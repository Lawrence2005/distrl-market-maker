# envs/ — Environment Layer

ABIDES-Gym extensions for this project.

## Files
- `lob_env.py`           — Base ABIDES-Gym market-making environment wrapper
- `stylized_facts.py`    — Post-episode stylized facts validator
- `multi_agent_env.py`   — Multi-agent wrapper (N simultaneous MM agents)
- `fine_grained.py`      — Sub-decision-interval recording: `FineGrainedRecorder`, per-second feature frames

Background agent population comes from ABIDES's own built-in `rmsc04`
config, unmodified — there is no project-specific background-agent module
here.

## Usage
```python
from envs.lob_env import LOBMarketMakingEnv
env = LOBMarketMakingEnv(config="configs/env/base.yaml")
obs, info = env.reset()
```

## Week 2 deliverable

## Market-data granularity and fine-grained recording

The agent decides once per `timestep_duration` (60 s), but ABIDES publishes
order-book snapshots and fills continuously in between. `MarketDataConfig`
(in `lob_env.py`, `env.market_data` in `training/configs/env/base.yaml`)
controls what the gym agent receives and records; its defaults are
ABIDES-Gym's own, so **leaving it alone changes nothing** about the policy's
observations or any existing result.

| field | meaning |
|---|---|
| `buffer_length` | L2 snapshots kept for the *policy* (default 5; ABIDES deep-copies this several times per step, so large values are slow) |
| `subscribe_freq_ns` | minimum gap between exchange L2 pushes (default 1e8 = 0.1 s); pushes are event-driven, in rmsc04 about one every ~5 s |
| `timestep_duration` | decision cadence (`null` = `"60s"`); changes the MDP, so rescale `episode_len` |
| `record` | expose every snapshot, volume push and timestamped fill in `info` (`book_events`, `volume_events`, `fill_events`, `decision_ts`, `mkt_open_ts`) via `RecordingGymAgent`; does not touch the simulation |

`FineGrainedRecorder` wraps a recording env, stores the raw event streams per
episode, and `frames(grid_s=...)` resamples them onto any grid — granularity
is an analysis-time choice, no re-simulation:

```python
env = LOBMarketMakingEnv(market_data=MarketDataConfig(record=True))
rec = FineGrainedRecorder(env)
obs, info = rec.reset(seed=90000)       # then rec.step(action) as usual
frames = rec.frames(grid_s=5.0)         # {"grid", "steps", "book", "volume", "fills"}
```

`grid` has one row per grid point carrying the 18 handcrafted features under
the env's own names/definitions (shared `book_features`, `compute_rsi`,
`compute_realized_vol`), plus quote/inventory state and reward on decision
rows. Inventory and the quote in force (`bid_offset`, `ask_offset`, `bid_dist`,
`ask_dist`) are continuous states, set on every row; `is_decision` marks where
the agent actually chose them, and the `steps` table has every decision.
Exchange-pushed data (book, volume) is forward-filled with an age column.
With `grid_s` equal to the decision interval, the rows at decision times
reproduce the observation the agent received (`tests/test_fine_grained.py`).
Windowed features (`vol`, `rsi`, `signed_vol`, `mid_move`) count grid points,
so at a finer grid they describe a shorter time span than the agent's.

On disk the event-level book and volume streams are one table: `book.parquet`
has a row per L2 snapshot with the volume pushed alongside it as
`mkt_bid_volume` / `mkt_ask_volume` (`merge_book_volume`); the in-memory
`frames()` still returns them as separate `book` and `volume` frames.

Collect to parquet with `scripts/collect_fine_grained.py` (policies: random,
fixedspread, as, glft); `--grid_s 5,60` (default) writes a 5 s grid and a 60 s
grid, the latter equal to the agent's observations row for row. Its default seeds avoid the held-out evaluation
block 80042–80056, which it refuses unless `--allow_held_out`.

Between decisions the agent does nothing: it wakes once per `timestep_duration`,
cancels and replaces its quotes then, and they rest until the next decision.
There is no per-second no-op action because there is no per-second wakeup.

Measured in rmsc04 (normal/high_vol, GLFT policy), which sets what a grid can
and cannot resolve:
- The exchange calls `publish_order_book_data()` after every limit order,
  market order and cancel, and pushes to a subscriber only if at least
  `subscribe_freq_ns` (0.1 s) has passed since its last push to that
  subscriber. So a gap between snapshots means no order events happened — the
  book was quiet — and forward-filling the latest snapshot is essentially
  exact (the one error: an event within 0.1 s of a push is dropped, not
  deferred). `book_age_s` is "time since the last order event".
- Snapshots arrive irregularly: ~11.7 per minute (mean gap ~5 s), median gap
  ~4 s, 10% under 1 s, 10% over 14 s, up to 60 s. Only ~1% repeat the
  previous snapshot. A 5 s grid therefore has a snapshot at least 5 s old on
  ~44% of rows; 5 s is the default and a finer grid only forward-fills more.
- ~12% of snapshots land within 1 s after a decision (uniform would be ~2%):
  the agent's own cancel-and-replace creates book updates, and its resting
  quotes are part of the depth/imbalance features — as in the agent's own
  observation.
- Market-wide volume (`mkt_bid_volume`, `mkt_ask_volume`) is a rolling
  trailing-60 s total pushed alongside each book snapshot, not per-interval
  flow.
- Windowed features (`vol`, `rsi`, `signed_vol`, `mid_move`) count grid
  points; `scripts/collect_fine_grained.py` takes windows in seconds so they
  span the agent's time at any grid. `vol` is the std of grid-step returns,
  so it scales with grid spacing (smaller at 5 s than at 60 s); `rsi` doesn't.
- The first row is market open + 30 s (ABIDES `first_interval`); see
  `t_since_open_s`.
- Rollouts reflect the policy that produced them (random/baseline quotes
  change fills and the book). Trained-agent checkpoints are not yet a
  `--policy` option.
