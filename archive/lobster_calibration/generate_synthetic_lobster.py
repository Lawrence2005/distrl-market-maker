"""
data/synthetic/generate_synthetic_lobster.py

Generates synthetic LOB data in LOBSTER format:
  - {symbol}_{date}_34200000_57600000_message_{levels}.csv
  - {symbol}_{date}_34200000_57600000_orderbook_{levels}.csv

Mid-price follows arithmetic Brownian motion (AS 2008 assumption).
Order arrivals follow a Hawkes process (Bacry et al. 2015).
Fill intensities follow exponential decay with distance (AS 2008).

Usage:
    python data/synthetic/generate_synthetic_lobster.py
    python data/synthetic/generate_synthetic_lobster.py --sigma 0.002 --n_levels 10
"""

import numpy as np
import pandas as pd
import argparse
import yaml
import os
from pathlib import Path


def generate_mid_price_path(
    S0: float,
    sigma: float,
    dt: float,
    n_steps: int,
    drift: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Arithmetic Brownian motion mid-price path.
    Matches the AS (2008) assumption of driftless BM (set drift=0.0).
    For trending regime tests, set drift != 0.

    Returns array of shape (n_steps,) with mid-prices in raw integer form
    (multiplied by 10000 to match LOBSTER format).
    """
    rng = np.random.default_rng(seed)
    increments = drift * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_steps)
    path = S0 + np.cumsum(increments)
    # Convert to LOBSTER integer format (× 10000), round to nearest tick
    tick_size = 100   # $0.01 in integer format
    path_int = np.round(path * 10000 / tick_size).astype(int) * tick_size
    return path_int


def simulate_hawkes_arrivals(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate Hawkes process arrivals via Ogata thinning.
    Returns array of event times in seconds.

    Parameters match Bacry et al. (2015):
        mu    — baseline intensity (events/sec)
        alpha — excitation magnitude
        beta  — decay rate; branching ratio rho = alpha/beta must be < 1
    """
    assert alpha / beta < 1.0, f"Branching ratio {alpha/beta:.3f} >= 1; not stationary"
    rng = np.random.default_rng(seed)

    times = []
    t = 0.0

    while t < T:
        # Upper bound on intensity
        if len(times) == 0:
            lam_upper = mu
        else:
            lam_upper = mu + alpha * np.sum(
                np.exp(-beta * (t - np.array(times)))
            )

        # Propose next event time
        dt = rng.exponential(1.0 / lam_upper)
        t_proposed = t + dt

        if t_proposed > T:
            break

        # Compute true intensity at proposed time
        if len(times) == 0:
            lam_true = mu
        else:
            lam_true = mu + alpha * np.sum(
                np.exp(-beta * (t_proposed - np.array(times)))
            )

        # Accept/reject
        if rng.uniform() < lam_true / lam_upper:
            times.append(t_proposed)

        t = t_proposed

    return np.array(times)


def build_lob_snapshot(
    mid: int,
    n_levels: int,
    tick_size: int = 100,
    base_size: int = 100,
    size_decay: float = 0.7,
    rng: np.random.Generator = None,
) -> dict:
    """
    Build a synthetic LOB snapshot around a given mid-price.

    Spreads quotes symmetrically around mid, with volumes decreasing
    geometrically away from the best quote.

    Returns dict with keys:
        ask_prices, ask_sizes, bid_prices, bid_sizes  (each length n_levels)
    """
    if rng is None:
        rng = np.random.default_rng()

    half_spread = tick_size  # 1-tick spread by default

    ask_prices = [mid + half_spread + i * tick_size for i in range(n_levels)]
    bid_prices = [mid - half_spread - i * tick_size for i in range(n_levels)]

    # Volumes: geometric decay with noise
    ask_sizes = [
        max(1, int(base_size * (size_decay ** i) + rng.integers(-10, 10)))
        for i in range(n_levels)
    ]
    bid_sizes = [
        max(1, int(base_size * (size_decay ** i) + rng.integers(-10, 10)))
        for i in range(n_levels)
    ]

    return {
        "ask_prices": ask_prices,
        "ask_sizes":  ask_sizes,
        "bid_prices": bid_prices,
        "bid_sizes":  bid_sizes,
    }


def generate_synthetic_lobster(
    symbol:     str   = "SYNTH",
    date:       str   = "2024-01-02",
    S0:         float = 150.0,
    sigma:      float = 0.001,
    drift:      float = 0.0,
    mu:         float = 0.5,
    alpha:      float = 0.3,
    beta:       float = 1.0,
    n_levels:   int   = 10,
    T_seconds:  float = 23400.0,   # 6.5 trading hours
    seed:       int   = 42,
    output_dir: str   = "data/synthetic/generated",
):
    """
    Full synthetic LOBSTER data generator.

    Produces two CSV files matching LOBSTER format exactly so that
    data/process_lobster.py works identically on synthetic and real data.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- 1. Simulate arrival times via Hawkes process ---
    print(f"Simulating Hawkes arrivals (mu={mu}, alpha={alpha}, beta={beta})...")
    arrival_times = simulate_hawkes_arrivals(mu, alpha, beta, T_seconds, seed=seed)
    n_events = len(arrival_times)
    print(f"  Generated {n_events} events over {T_seconds:.0f} seconds")

    # Convert to absolute timestamps (market opens at 9:30 AM = 34200 seconds)
    timestamps = arrival_times + 34200.0

    # --- 2. Generate mid-price path at each event time ---
    dt_avg = T_seconds / n_events
    mid_prices = generate_mid_price_path(S0, sigma, dt_avg, n_events, drift, seed)

    # --- 3. Build message file ---
    # Alternate between limit order submissions (type 1) and
    # executions (type 4) to produce a realistic event stream
    message_rows = []
    orderbook_rows = []

    order_id = 10000000
    tick_size = 100   # $0.01 in integer format

    for i, (ts, mid) in enumerate(zip(timestamps, mid_prices)):
        snap = build_lob_snapshot(mid, n_levels, tick_size, rng=rng)

        # Randomly assign event type: 70% new orders, 20% cancellations, 10% executions
        r = rng.uniform()
        if r < 0.70:
            ev_type = 1   # new limit order
        elif r < 0.90:
            ev_type = 3   # full cancellation
        else:
            ev_type = 4   # execution

        direction = 1 if rng.uniform() < 0.5 else -1
        size = int(rng.integers(1, 5) * 100)

        if direction == 1:
            price = snap["bid_prices"][0]
        else:
            price = snap["ask_prices"][0]

        message_rows.append([ts, ev_type, order_id, size, price, direction])
        order_id += 1

        # Orderbook row: interleave ask/bid at each level
        ob_row = []
        for lvl in range(n_levels):
            ob_row += [snap["ask_prices"][lvl], snap["ask_sizes"][lvl],
                       snap["bid_prices"][lvl], snap["bid_sizes"][lvl]]
        orderbook_rows.append(ob_row)

    # --- 4. Save to CSV in LOBSTER format ---
    date_str = date.replace("-", "")
    base = f"{symbol}_{date_str}_34200000_57600000"

    msg_path = os.path.join(output_dir, f"{base}_message_{n_levels}.csv")
    ob_path  = os.path.join(output_dir, f"{base}_orderbook_{n_levels}.csv")

    # Message file: no header (LOBSTER convention)
    msg_df = pd.DataFrame(
        message_rows,
        columns=["Time", "Type", "Order ID", "Size", "Price", "Direction"]
    )
    msg_df.to_csv(msg_path, index=False, header=False)

    # Orderbook file: no header
    ob_cols = []
    for lvl in range(1, n_levels + 1):
        ob_cols += [f"Ask Price {lvl}", f"Ask Size {lvl}",
                    f"Bid Price {lvl}", f"Bid Size {lvl}"]
    ob_df = pd.DataFrame(orderbook_rows, columns=ob_cols)
    ob_df.to_csv(ob_path, index=False, header=False)

    print(f"  Saved message file:   {msg_path}")
    print(f"  Saved orderbook file: {ob_path}")
    print(f"  Rows: {len(msg_df)} (message) = {len(ob_df)} (orderbook) ✓")

    return msg_path, ob_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="SYNTH")
    parser.add_argument("--date",      default="2024-01-02")
    parser.add_argument("--S0",        type=float, default=150.0)
    parser.add_argument("--sigma",     type=float, default=0.001)
    parser.add_argument("--drift",     type=float, default=0.0)
    parser.add_argument("--mu",        type=float, default=0.5)
    parser.add_argument("--alpha",     type=float, default=0.3)
    parser.add_argument("--beta",      type=float, default=1.0)
    parser.add_argument("--n_levels",  type=int,   default=10)
    parser.add_argument("--T_seconds", type=float, default=23400.0)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    generate_synthetic_lobster(
        symbol=args.symbol, date=args.date,
        S0=args.S0, sigma=args.sigma, drift=args.drift,
        mu=args.mu, alpha=args.alpha, beta=args.beta,
        n_levels=args.n_levels, T_seconds=args.T_seconds,
        seed=args.seed,
    )