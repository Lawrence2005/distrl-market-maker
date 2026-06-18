"""
LOBSTER data processing script.

Reads raw LOBSTER message + orderbook files.
Outputs:
  - LOB snapshot tensors for AE pre-training
  - Background agent calibration parameters (hawkes_params.json)
  - Background agent parameters (agent_params.json)
  - Stylized facts summary for simulator validation

Week 2 deliverable.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path


def fit_hawkes_mle(
    times: np.ndarray,
    mu0: float = 0.5,
    alpha0: float = 0.3,
    beta0: float = 1.0,
) -> dict:
    """
    Fit Hawkes process parameters (mu, alpha, beta) via Maximum Likelihood
    Estimation on a sequence of observed event arrival times.

    The log-likelihood of a Hawkes process with exponential kernel is:

        log L = -mu*T
                - alpha/beta * sum_i (1 - exp(-beta*(T - t_i)))
                + sum_i log(mu + alpha * sum_{j: t_j < t_i} exp(-beta*(t_i - t_j)))

    We maximise this using scipy.optimize.minimize with L-BFGS-B, which
    handles the box constraints mu>0, alpha>0, beta>0 naturally.

    Parameters
    ----------
    times  : np.ndarray — sorted array of event arrival times in seconds
    mu0    : float      — initial guess for baseline intensity
    alpha0 : float      — initial guess for excitation magnitude
    beta0  : float      — initial guess for decay rate

    Returns
    -------
    dict with keys: mu, alpha, beta, branching_ratio
    """
    from scipy.optimize import minimize

    T = times[-1]   # total observation window
    n = len(times)

    def neg_log_likelihood(params):
        mu, alpha, beta = params
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e10

        # Branching ratio must be < 1 for stationarity
        if alpha / beta >= 1.0:
            return 1e10

        # Recursive computation of A_i = sum_{j<i} exp(-beta*(t_i - t_j))
        # using the recurrence: A_i = exp(-beta*(t_i - t_{i-1})) * (1 + A_{i-1})
        # This is O(n) rather than O(n^2)
        A = np.zeros(n)
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            A[i] = np.exp(-beta * dt) * (1 + A[i - 1])

        # Log-likelihood components
        term1 = -mu * T
        term2 = -(alpha / beta) * np.sum(1 - np.exp(-beta * (T - times)))
        term3 = np.sum(np.log(mu + alpha * A))

        return -(term1 + term2 + term3)

    result = minimize(
        neg_log_likelihood,
        x0=[mu0, alpha0, beta0],
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    mu_hat, alpha_hat, beta_hat = result.x
    branching_ratio = alpha_hat / beta_hat

    print(f"  Hawkes MLE converged: {result.success}")
    print(f"  mu={mu_hat:.4f}, alpha={alpha_hat:.4f}, beta={beta_hat:.4f}")
    print(f"  Branching ratio rho = alpha/beta = {branching_ratio:.4f} (must be <1)")

    if branching_ratio >= 1.0:
        print("  WARNING: branching ratio >= 1 — process is not stationary.")
        print("  Consider re-running with tighter bounds or more data.")

    return {
        "mu":              float(mu_hat),
        "alpha":           float(alpha_hat),
        "beta":            float(beta_hat),
        "branching_ratio": float(branching_ratio),
        "n_events":        int(n),
        "T_seconds":       float(T),
        "converged":       bool(result.success),
    }


def compute_agent_params(all_messages: pd.DataFrame) -> dict:
    """
    Compute background agent calibration parameters from message file data.

    These parameters are used to configure the three background agent types
    in ABIDES-Gym (noise, momentum, informed) so their behaviour matches
    the empirical distribution of order flow observed in the data.

    Parameters derived:
      - arrival_rate_per_sec : mean number of order events per second
                               (used to set noise trader Poisson rate)
      - mean_order_size      : mean number of shares per order
      - std_order_size       : standard deviation of order sizes
      - cancellation_rate    : fraction of events that are cancellations
                               (type 2 or 3) — used to set cancel probability
      - buy_sell_ratio       : fraction of events on the buy side (direction=1)
                               vs sell side (direction=-1)
      - mean_interarrival_sec: mean time between consecutive events in seconds

    Parameters
    ----------
    all_messages : pd.DataFrame — concatenated message file rows across all
                                  processed files, with columns:
                                  Time, Type, OrderID, Size, Price, Direction

    Returns
    -------
    dict of calibration parameters consumed by envs/background_agents.py
    """
    total_time = all_messages["Time"].max() - all_messages["Time"].min()
    n_events   = len(all_messages)

    # Arrival rate
    arrival_rate = n_events / total_time if total_time > 0 else 1.0

    # Order size distribution
    mean_size = float(all_messages["Size"].mean())
    std_size  = float(all_messages["Size"].std())
    min_size  = int(all_messages["Size"].min())
    max_size  = int(all_messages["Size"].max())

    # Cancellation rate: event types 2 (partial cancel) and 3 (full cancel)
    cancel_mask      = all_messages["Type"].isin([2, 3])
    cancellation_rate = float(cancel_mask.sum() / n_events)

    # Execution rate: event types 4 and 5
    exec_mask      = all_messages["Type"].isin([4, 5])
    execution_rate = float(exec_mask.sum() / n_events)

    # Buy/sell ratio
    buy_mask      = all_messages["Direction"] == 1
    buy_sell_ratio = float(buy_mask.sum() / n_events)

    # Interarrival times
    interarrival = all_messages["Time"].diff().dropna()
    mean_interarrival = float(interarrival.mean())
    std_interarrival  = float(interarrival.std())

    params = {
        # Used by NoiseAgent in envs/background_agents.py
        "arrival_rate_per_sec":  float(arrival_rate),
        "mean_interarrival_sec": mean_interarrival,
        "std_interarrival_sec":  std_interarrival,

        # Used by all agent types for order sizing
        "mean_order_size": mean_size,
        "std_order_size":  std_size,
        "min_order_size":  min_size,
        "max_order_size":  max_size,

        # Used by NoiseAgent and MomentumAgent
        "cancellation_rate": cancellation_rate,
        "execution_rate":    execution_rate,
        "buy_sell_ratio":    buy_sell_ratio,

        # Metadata
        "n_events":      n_events,
        "total_time_sec": float(total_time),
    }

    print(f"  Arrival rate:     {arrival_rate:.3f} events/sec")
    print(f"  Mean order size:  {mean_size:.1f} shares")
    print(f"  Cancel rate:      {cancellation_rate:.3f}")
    print(f"  Buy/sell ratio:   {buy_sell_ratio:.3f}")

    return params


def process_lobster_directory(
    data_dir:         str,
    n_levels:         int = 10,
    output_snapshots: str = "data/processed/lob_snapshots.npy",
    output_hawkes:    str = "data/calibration/hawkes_params.json",
    output_agents:    str = "data/calibration/agent_params.json",
) -> tuple[np.ndarray, dict, dict]:
    """
    Reads all message + orderbook CSV pairs in data_dir.
    Works identically on:
        data/synthetic/generated/   ← synthetic data
        data/crypto/raw/            ← Binance crypto data
        data/lobster/               ← real LOBSTER equity data
    """
    # Ensure output directories exist
    Path(output_snapshots).parent.mkdir(parents=True, exist_ok=True)
    Path(output_hawkes).parent.mkdir(parents=True, exist_ok=True)
    Path(output_agents).parent.mkdir(parents=True, exist_ok=True)

    # Find all message files
    message_files = sorted(Path(data_dir).glob(f"*_message_{n_levels}.csv"))

    if len(message_files) == 0:
        raise FileNotFoundError(
            f"No message files found in {data_dir} matching "
            f"*_message_{n_levels}.csv. "
            f"Run data/synthetic/generate_synthetic_lobster.py first, or "
            f"check that your n_levels={n_levels} matches the files."
        )

    print(f"Found {len(message_files)} message file(s) in {data_dir}")

    all_snapshots  = []
    hawkes_times   = []
    all_messages   = []

    for msg_path in message_files:
        ob_path = str(msg_path).replace("_message_", "_orderbook_")

        if not Path(ob_path).exists():
            print(f"  WARNING: no matching orderbook file for {msg_path.name} — skipping")
            continue

        print(f"  Processing {msg_path.name}...")

        # Read files (no header — LOBSTER convention)
        msg = pd.read_csv(
            msg_path, header=None,
            names=["Time", "Type", "OrderID", "Size", "Price", "Direction"]
        )
        ob = pd.read_csv(ob_path, header=None)

        # Prices: divide by 10000 to get dollars
        msg["Price"] = msg["Price"] / 10000.0

        price_cols = list(range(0, ob.shape[1], 2))   # columns 0, 2, 4, ... are prices
        ob.iloc[:, price_cols] = ob.iloc[:, price_cols] / 10000.0

        # ── Extract LOB snapshots for AE pre-training ──────────────────
        # Each row of the orderbook file is one snapshot of the LOB state.
        # We build a flat vector of [ask_sizes..., bid_sizes...] (2K dims)
        # because the AE learns to compress the depth profile shape,
        # not the absolute price levels.
        ask_size_cols = list(range(1, ob.shape[1], 4))   # columns 1, 5, 9, ...
        bid_size_cols = list(range(3, ob.shape[1], 4))   # columns 3, 7, 11, ...

        snapshots = np.concatenate([
            ob.iloc[:, ask_size_cols].values,
            ob.iloc[:, bid_size_cols].values,
        ], axis=1).astype(np.float32)

        all_snapshots.append(snapshots)

        # ── Collect timestamps for Hawkes calibration ──────────────────
        hawkes_times.extend(msg["Time"].tolist())

        # ── Collect message rows for agent parameter calibration ────────
        all_messages.append(msg)

    # ── Save LOB snapshots ──────────────────────────────────────────────
    snapshots_arr = np.vstack(all_snapshots)
    np.save(output_snapshots, snapshots_arr)
    print(f"\nSaved {len(snapshots_arr)} LOB snapshots → {output_snapshots}")
    print(f"  Snapshot shape: {snapshots_arr.shape}  "
          f"(each row = {snapshots_arr.shape[1]}-dim depth profile)")

    # ── Fit and save Hawkes parameters ──────────────────────────────────
    print("\nFitting Hawkes process via MLE...")
    hawkes_params = fit_hawkes_mle(np.array(sorted(hawkes_times)))
    with open(output_hawkes, "w") as f:
        json.dump(hawkes_params, f, indent=2)
    print(f"Saved Hawkes params → {output_hawkes}")

    # ── Compute and save agent calibration parameters ───────────────────
    print("\nComputing background agent calibration parameters...")
    combined_messages = pd.concat(all_messages, ignore_index=True)
    agent_params = compute_agent_params(combined_messages)
    with open(output_agents, "w") as f:
        json.dump(agent_params, f, indent=2)
    print(f"Saved agent params → {output_agents}")

    return snapshots_arr, hawkes_params, agent_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process LOB data (synthetic, crypto, or real LOBSTER) "
                    "into calibration parameters and AE pre-training snapshots."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/synthetic/generated",
        help="Path to directory containing message + orderbook CSV pairs. "
             "Works with: data/synthetic/generated/, data/crypto/raw/, data/lobster/"
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=10,
        help="Number of LOB depth levels in the CSV files "
             "(must match what was generated)."
    )
    parser.add_argument(
        "--output_snapshots",
        type=str,
        default="data/processed/lob_snapshots.npy",
        help="Where to save the processed LOB snapshot array for AE pre-training."
    )
    parser.add_argument(
        "--output_hawkes",
        type=str,
        default="data/calibration/hawkes_params.json",
        help="Where to save the fitted Hawkes parameters JSON."
    )
    parser.add_argument(
        "--output_agents",
        type=str,
        default="data/calibration/agent_params.json",
        help="Where to save the background agent calibration parameters JSON."
    )
    args = parser.parse_args()

    snapshots, hawkes_params, agent_params = process_lobster_directory(
        data_dir=args.data_dir,
        n_levels=args.n_levels,
        output_snapshots=args.output_snapshots,
        output_hawkes=args.output_hawkes,
        output_agents=args.output_agents,
    )

    print("\n=== process_lobster.py complete ===")
    print(f"  LOB snapshots : {snapshots.shape} → {args.output_snapshots}")
    print(f"  Hawkes params : mu={hawkes_params['mu']:.4f}, "
          f"alpha={hawkes_params['alpha']:.4f}, "
          f"beta={hawkes_params['beta']:.4f}, "
          f"rho={hawkes_params['branching_ratio']:.4f}")
    print(f"  Agent params  : arrival_rate={agent_params['arrival_rate_per_sec']:.3f} "
          f"events/sec, mean_size={agent_params['mean_order_size']:.1f}")
    print(f"  Outputs saved : {args.output_hawkes}, {args.output_agents}")