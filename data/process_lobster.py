"""
LOBSTER data processing script.

Reads raw LOBSTER message + orderbook files.
Outputs:
  - LOB snapshot tensors for AE pre-training
  - Stylized facts summary for simulator validation

Week 2 deliverable.

(This project's own background-agent calibration pipeline — which used to
compute NoiseAgent/MomentumAgent/InformedAgent parameters from this same
message data — was archived: those custom agent classes were never wired
into the live simulator, which uses ABIDES's own rmsc04 background config
unmodified. See `archive/lobster_calibration/` if reviving that pipeline.)
"""
import pandas as pd
import numpy as np
from pathlib import Path

def process_lobster_directory(
    data_dir:         str,
    n_levels:         int = 10,
    output_snapshots: str = "data/processed/lob_snapshots.npy",
) -> np.ndarray:
    """
    Reads all message + orderbook CSV pairs in data_dir.
    Works identically on:
        data/crypto/raw/            ← Binance crypto data
        data/lobster/               ← real LOBSTER equity data
    """
    # Ensure output directory exists
    Path(output_snapshots).parent.mkdir(parents=True, exist_ok=True)

    # Find all message files
    message_files = sorted(Path(data_dir).glob(f"*_message_{n_levels}.csv"))

    if not message_files:
        raise FileNotFoundError(
            f"No message files found in {data_dir} matching "
            f"*_message_{n_levels}.csv. "
            f"Run data/crypto/fetch_binance_lob.py first, or place real "
            f"LOBSTER files in data/lobster/, or check that your "
            f"n_levels={n_levels} matches the files."
        )

    print(f"Found {len(message_files)} message file(s) in {data_dir}")

    all_snapshots  = []

    for msg_path in message_files:
        ob_path = str(msg_path).replace("_message_", "_orderbook_")

        if not Path(ob_path).exists():
            print(f"  WARNING: no matching orderbook file for {msg_path.name} — skipping")
            continue

        print(f"  Processing {msg_path.name}...")

        # Read the orderbook file (no header — LOBSTER convention). The
        # paired message file only ever supplied ob_path's glob match above
        # — nothing downstream (snapshot extraction) reads message-level
        # data, so it was never loaded here (a prior version read it into
        # `msg` and rescaled its Price column, then discarded the result
        # without using it anywhere — dead computation, removed).
        ob = pd.read_csv(ob_path, header=None)

        # Prices: divide by 10000 to get dollars
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

    if not all_snapshots:
        raise FileNotFoundError(
            "No valid message/orderbook pairs were processed."
        )

    # ── Save LOB snapshots ──────────────────────────────────────────────
    snapshots_arr = np.vstack(all_snapshots)
    np.save(output_snapshots, snapshots_arr)
    print(f"\nSaved {len(snapshots_arr)} LOB snapshots → {output_snapshots}")
    print(f"  Snapshot shape: {snapshots_arr.shape}  "
          f"(each row = {snapshots_arr.shape[1]}-dim depth profile)")

    return snapshots_arr

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process LOB data (crypto or real LOBSTER) into "
                    "AE pre-training snapshots."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/crypto/raw",
        help="Path to directory containing message + orderbook CSV pairs. "
             "Works with: data/crypto/raw/, data/lobster/"
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
    args = parser.parse_args()

    snapshots = process_lobster_directory(
        data_dir=args.data_dir,
        n_levels=args.n_levels,
        output_snapshots=args.output_snapshots,
    )

    print("\n=== process_lobster.py complete ===")
    print(f"  LOB snapshots : {snapshots.shape} → {args.output_snapshots}")