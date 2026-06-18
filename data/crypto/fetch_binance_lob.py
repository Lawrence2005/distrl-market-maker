"""
data/crypto/fetch_binance_lob.py

Fetches real LOB snapshots from Binance public API and saves them
in LOBSTER-compatible format for use with process_lobster.py.

No API key required — Binance LOB snapshots are public.

Usage:
    python data/crypto/fetch_binance_lob.py
    python data/crypto/fetch_binance_lob.py --symbol BTCUSDT --n_levels 10 --n_snapshots 5000
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import argparse
from pathlib import Path
from datetime import datetime


def fetch_binance_orderbook(
    symbol: str = "BTCUSDT",
    limit:  int = 20,
) -> dict:
    """
    Fetch a single LOB snapshot from Binance REST API.

    Returns dict with bids and asks, each a list of [price, quantity] pairs
    sorted by price (asks ascending, bids descending).
    """
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {
        "timestamp": time.time(),
        "asks": [[float(p), float(q)] for p, q in data["asks"]],
        "bids": [[float(p), float(q)] for p, q in data["bids"]],
    }


def fetch_and_save_lobster_format(
    symbol:       str = "BTCUSDT",
    n_levels:     int = 10,
    n_snapshots:  int = 5000,
    interval_sec: float = 0.5,
    output_dir:   str = "data/crypto/raw",
    scale_price:  bool = True,
):
    """
    Repeatedly fetch LOB snapshots and save in LOBSTER format.

    Parameters
    ----------
    symbol       : Binance trading pair, e.g. "BTCUSDT", "ETHUSDT"
    n_levels     : number of LOB depth levels to record
    n_snapshots  : total number of snapshots to collect
    interval_sec : pause between fetches (seconds); 0.5 = ~2 per second
    scale_price  : if True, scale crypto prices to equity-like range ($100–200)
                   by normalising to the first snapshot's mid-price × 150
    output_dir   : where to save output CSVs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    base     = f"{symbol}_{date_str}_34200000_57600000"

    message_rows  = []
    orderbook_rows = []
    order_id      = 10000000
    first_mid     = None
    scale_factor  = 1.0

    print(f"Fetching {n_snapshots} LOB snapshots for {symbol}...")

    for i in range(n_snapshots):
        try:
            snap = fetch_binance_orderbook(symbol, limit=n_levels + 5)
        except Exception as e:
            print(f"  Fetch {i} failed: {e}. Retrying...")
            time.sleep(2)
            continue

        asks = snap["asks"][:n_levels]
        bids = snap["bids"][:n_levels]

        if len(asks) < n_levels or len(bids) < n_levels:
            continue

        # Compute mid-price
        best_ask = asks[0][0]
        best_bid = bids[0][0]
        mid      = (best_ask + best_bid) / 2

        # Scale to equity-like prices if requested
        if scale_price:
            if first_mid is None:
                first_mid    = mid
                scale_factor = 150.0 / first_mid
            asks = [[p * scale_factor, q] for p, q in asks]
            bids = [[p * scale_factor, q] for p, q in bids]
            mid  = mid * scale_factor

        # Convert to LOBSTER integer format (× 10000)
        tick = 100   # $0.01
        to_int = lambda p: int(round(p * 10000 / tick) * tick)

        ask_prices = [to_int(p) for p, q in asks]
        ask_sizes  = [max(1, int(q * 100)) for p, q in asks]  # normalise qty
        bid_prices = [to_int(p) for p, q in bids]
        bid_sizes  = [max(1, int(q * 100)) for p, q in bids]

        # Timestamp: convert to seconds since midnight (offset from 9:30 AM)
        ts = 34200.0 + i * interval_sec

        # Message row: treat each snapshot as a new limit order submission
        price_int = ask_prices[0]
        message_rows.append([
            ts, 1, order_id, ask_sizes[0], price_int, -1
        ])
        order_id += 1

        # Orderbook row
        ob_row = []
        for lvl in range(n_levels):
            ob_row += [ask_prices[lvl], ask_sizes[lvl],
                       bid_prices[lvl], bid_sizes[lvl]]
        orderbook_rows.append(ob_row)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_snapshots} snapshots collected")

        time.sleep(interval_sec)

    # Save in LOBSTER format
    msg_path = os.path.join(output_dir, f"{base}_message_{n_levels}.csv")
    ob_path  = os.path.join(output_dir, f"{base}_orderbook_{n_levels}.csv")

    pd.DataFrame(message_rows).to_csv(msg_path, index=False, header=False)

    ob_cols = []
    for lvl in range(1, n_levels + 1):
        ob_cols += [f"Ask Price {lvl}", f"Ask Size {lvl}",
                    f"Bid Price {lvl}", f"Bid Size {lvl}"]
    pd.DataFrame(orderbook_rows, columns=ob_cols).to_csv(
        ob_path, index=False, header=False
    )

    print(f"\nSaved {len(message_rows)} rows to:")
    print(f"  {msg_path}")
    print(f"  {ob_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",      default="BTCUSDT")
    parser.add_argument("--n_levels",    type=int,   default=10)
    parser.add_argument("--n_snapshots", type=int,   default=5000)
    parser.add_argument("--interval",    type=float, default=0.5)
    args = parser.parse_args()

    fetch_and_save_lobster_format(
        symbol=args.symbol,
        n_levels=args.n_levels,
        n_snapshots=args.n_snapshots,
        interval_sec=args.interval,
    )