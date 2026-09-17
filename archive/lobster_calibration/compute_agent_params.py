"""
archive/lobster_calibration/compute_agent_params.py

Archived from data/process_lobster.py. Computes background-agent calibration
parameters from LOBSTER-format message data, for configuring the custom
NoiseAgent/MomentumAgent/InformedAgent classes in
archive/lobster_calibration/background_agents.py.

This pipeline was never wired into the live simulator — the real ABIDES
path (`envs/lob_env.py`'s `build_abides_env`, `use_abides=True`) uses
ABIDES's own rmsc04 background-agent config unmodified, so these calibrated
parameters were never actually consumed by any training run in this
project. Archived rather than deleted in case a future revival is wanted.

See also archive/lobster_calibration/generate_synthetic_lobster.py (the
synthetic-data generator this was originally calibrated against) and
archive/lobster_calibration/agent_params.json (the last computed output).
"""
import pandas as pd


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
    dict of calibration parameters consumed by
    archive/lobster_calibration/background_agents.py
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
        # Used by NoiseAgent in background_agents.py
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
