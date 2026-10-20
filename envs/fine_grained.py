"""
Fine-grained (sub-decision-interval) recording and feature frames.

The agent decides once per `timestep_duration` (60 s), but ABIDES publishes
order-book snapshots and fills continuously in between. This module keeps
all of it and resamples it onto whatever time grid an analysis wants,
without touching the simulation or the policy's observations:

    env = LOBMarketMakingEnv(market_data=MarketDataConfig(record=True))
    rec = FineGrainedRecorder(env)
    obs, info = rec.reset(seed=s)          # then rec.step(action) as usual
    frames = rec.frames(grid_s=1.0)        # {"grid", "steps", "book", "fills"}

The raw record is lossless and event-level (snapshots arrive every few
seconds, event-driven — not on a clock); the grid is an analysis-time
choice, so changing granularity (1 s, 0.1 s, 5 s, ...) needs no re-simulation.

`grid` has one row per grid point, forward-filling the latest snapshot, with
the handcrafted-observation features under the same names and definitions as
LOBMarketMakingEnv._get_obs (shared via lob_env.book_features /
compute_rsi / compute_realized_vol). With `grid_s` equal to the decision
interval, the rows at decision times reproduce the observation the agent saw.
"""
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from envs.lob_env import (
    TICK_OFFSETS,
    _PRICE_HISTORY_MAXLEN,
    _VOLUME_HISTORY_MAXLEN,
    book_features,
    compute_realized_vol,
    compute_rsi,
)

_NS = 1_000_000_000

# Per-step columns recorded from `info` (the agent's decision and its outcome)
_STEP_FIELDS = (
    "step", "decision_ts", "bid_offset", "ask_offset", "bid_price", "ask_price",
    "reward", "inventory", "cash", "mid_price", "bid_filled", "ask_filled",
    "spread_pnl",
)


def feature_columns(n_lob_levels: int = 3) -> List[str]:
    """
    Names of the handcrafted-observation features, in observation order
    (docs/mdp_formulation.md §2; index layout in LOBMarketMakingEnv._get_obs).
    """
    return (
        ["spread", "mid_move", "imbalance", "signed_vol", "vol", "rsi"]
        + [f"bid_depth_{i + 1}" for i in range(n_lob_levels)]
        + [f"ask_depth_{i + 1}" for i in range(n_lob_levels)]
        + ["inventory", "bid_dist", "ask_dist",
           "outstanding_bid", "outstanding_ask", "time_remaining"]
    )


class FineGrainedRecorder:
    """
    Wraps a LOBMarketMakingEnv built with `MarketDataConfig(record=True)`,
    passes reset()/step() straight through, and stores per episode the raw
    book events, timestamped fills and per-decision rows.

    A plain delegating wrapper (not gym.Wrapper): scripts keep using the
    env's own attributes (Q_max, tick_size, ...) via attribute passthrough,
    and the wrapped env's observations, rewards and info are returned
    unchanged.
    """

    def __init__(self, env):
        if not getattr(env.market_data, "record", False) or env._abides_env is None:
            raise ValueError(
                "FineGrainedRecorder needs a real-ABIDES env built with "
                "MarketDataConfig(record=True)."
            )
        self.env = env
        self._episodes: List[Dict[str, Any]] = []

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._episodes.append({
            "seed":  seed,
            "t0":       info["decision_ts"],
            "mkt_open": info["mkt_open_ts"],
            "book":     list(info["book_events"]),
            "volume":   list(info["volume_events"]),
            "fills":    list(info["fill_events"]),
            "steps":    [],
        })
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ep = self._episodes[-1]
        ep["book"].extend(info["book_events"])
        ep["volume"].extend(info["volume_events"])
        ep["fills"].extend(info["fill_events"])
        ep["steps"].append({k: info[k] for k in _STEP_FIELDS if k in info})
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def frames(
        self,
        grid_s: float = 1.0,
        vol_window: int = 20,
        rsi_window: int = 14,
    ) -> Dict[str, pd.DataFrame]:
        """
        All recorded episodes as DataFrames, each with `episode` and `seed`
        columns:

            grid   one row per grid point (see build_grid_frame)
            steps  one row per decision: action, reward, inventory, ...
            book   one row per L2 snapshot (ts, mid, L1..L10 prices/sizes)
            volume one row per transacted-volume push (ts, bid/ask volume)
            fills  one row per own fill (ts, side, qty, price)
        """
        env = self.env
        grids, steps, books, volumes, fills = [], [], [], [], []
        for i, ep in enumerate(self._episodes):
            tag = {"episode": i, "seed": ep["seed"]}
            grids.append(build_grid_frame(
                ep, grid_s=grid_s, tick_size=env.tick_size,
                n_lob_levels=env.n_lob_levels, Q_max=env.Q_max,
                episode_len=env.episode_len,
                vol_window=vol_window, rsi_window=rsi_window,
            ).assign(**tag))
            steps.append(pd.DataFrame(ep["steps"]).assign(**tag))
            books.append(_book_frame(ep["book"]).assign(**tag))
            volumes.append(pd.DataFrame(
                ep["volume"], columns=["ts", "bid_volume", "ask_volume"],
            ).assign(**tag))
            fills.append(pd.DataFrame(
                ep["fills"], columns=["ts", "side", "quantity", "price_cents"],
            ).assign(**tag))
        return {
            "grid":  pd.concat(grids,  ignore_index=True),
            "steps": pd.concat(steps,  ignore_index=True),
            "book":  pd.concat(books,  ignore_index=True),
            "volume": pd.concat(volumes, ignore_index=True),
            "fills": pd.concat(fills,  ignore_index=True),
        }


def _book_frame(events: List[dict]) -> pd.DataFrame:
    """Flat per-snapshot table: ts, mid, last_transaction, then
    bid_price_i / bid_size_i / ask_price_i / ask_size_i per depth level."""
    rows = []
    for e in events:
        row = {"ts": e["ts"], "mid": e["mid"], "last_transaction": e["last_transaction"]}
        for side in ("bid", "ask"):
            for i, (p, q) in enumerate(zip(e[f"{side}_prices"], e[f"{side}_sizes"])):
                row[f"{side}_price_{i + 1}"] = p
                row[f"{side}_size_{i + 1}"]  = q
        rows.append(row)
    return pd.DataFrame(rows)


def merge_book_volume(book: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """One row per L2 snapshot with the volume pushed alongside it.

    The exchange sends a transacted-volume update with every book snapshot, so
    the two event streams have identical timestamps row for row; the trailing-60 s
    totals are appended to the book as `mkt_bid_volume` / `mkt_ask_volume` (the
    names the grid frame uses for the same quantity).
    """
    if len(book) != len(volume) or not (book["ts"].to_numpy() == volume["ts"].to_numpy()).all():
        raise ValueError("book and volume events are not paired one-to-one by ts")
    return book.assign(mkt_bid_volume=volume["bid_volume"].to_numpy(),
                       mkt_ask_volume=volume["ask_volume"].to_numpy())


def build_grid_frame(
    episode: Dict[str, Any],
    *,
    grid_s: float,
    tick_size: float,
    n_lob_levels: int,
    Q_max: int,
    episode_len: int,
    vol_window: int = 20,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """
    Resample one recorded episode onto a regular time grid.

    episode : {"t0": reset decision time (ns), "book": format_book_events()
        dicts, "fills": (ts, side, qty, price_cents) tuples, "steps": per-
        decision rows (_STEP_FIELDS), optionally "volume":
        format_volume_events() dicts and "mkt_open": market open (ns)} —
        what FineGrainedRecorder stores.
    grid_s  : grid spacing in seconds, anchored at t0 and running to the
        last decision. Pure function of the record, so any spacing can be
        built from one simulation.

    Each row describes the state at grid time t:
      * the book is the latest snapshot at or before t (forward-filled;
        `book_age_s` says how stale), so the spread/imbalance/depth features
        are what the market looked like then;
      * inventory = the env's post-step inventory at the last decision at or
        before t, plus fills since then;
      * the quote entries are the quotes resting at t — the action chosen at
        the preceding decision, held until the next one;
      * inventory and the quote in force (`bid_offset`, `ask_offset`,
        `bid_dist`, `ask_dist`) are continuous states, set on every row. The
        reset row has no quote yet (offsets NaN, distances 0 as in the
        agent's reset observation). `is_decision` marks the rows where the
        agent actually chose it; `outstanding_bid/ask` give the resting
        quote's offset from the *moving* mid;
      * mid_move, vol, rsi, signed_vol are computed over *grid points*
        (`vol_window` / `rsi_window` points back), the same functions as the
        env applies over decision steps. At `grid_s` = the decision
        interval these windows coincide with the agent's.
    Features carry the env's normalisation and clipping, so rows at decision
    times equal the observation the agent received (see tests).

    Extra columns beyond the 18 features: raw book state (`mid`, `best_bid`,
    `best_ask`, `book_age_s`, `n_book_updates`), `t_since_open_s`, own fills
    and quotes, and market-wide order flow `mkt_bid_volume` /
    `mkt_ask_volume` — the latest push of the trailing-lookback (60 s by
    default) transacted volume, forward-filled, with `volume_age_s` (NaN
    until the first push). These are rolling totals, not per-interval flow.
    """
    book, steps = episode["book"], episode["steps"]
    t0 = int(episode["t0"])
    marks = np.array([t0] + [int(r["decision_ts"]) for r in steps], dtype=np.int64)
    interval = int(marks[1] - marks[0]) if len(marks) > 1 else 0
    grid_ns = int(round(grid_s * _NS))
    if grid_ns <= 0:
        raise ValueError(f"grid_s must be positive, got {grid_s}")

    ev_ts = np.array([e["ts"] for e in book], dtype=np.int64)
    if len(ev_ts) == 0:
        raise ValueError("episode has no recorded book events")

    ts = np.arange(t0, marks[-1] + 1, grid_ns, dtype=np.int64)
    ts = ts[ts >= ev_ts[0]]                 # no book to describe before this
    n = len(ts)

    # ── Per-snapshot features, computed once, then forward-filled ────────
    ev_feat = [book_features(e, tick_size, n_lob_levels) for e in book]
    ev_spread = np.array([f[0] for f in ev_feat])
    ev_imbal  = np.array([f[1] for f in ev_feat])
    ev_bid_d  = np.array([f[2] for f in ev_feat])
    ev_ask_d  = np.array([f[3] for f in ev_feat])
    ev_mid    = np.array([e["mid"] for e in book])
    ev_best_bid = np.array([e["bid_prices"][0] if e["bid_prices"] else np.nan for e in book])
    ev_best_ask = np.array([e["ask_prices"][0] if e["ask_prices"] else np.nan for e in book])

    idx   = np.searchsorted(ev_ts, ts, side="right") - 1
    mid   = np.maximum(ev_mid[idx], tick_size)        # env keeps mid >= tick
    prev  = np.concatenate([[-1], ts[:-1]])           # exclusive lower bound of each row's interval
    n_book = np.searchsorted(ev_ts, ts, side="right") - np.searchsorted(ev_ts, prev, side="right")

    # ── Fills: per-interval shares and running inventory ─────────────────
    fills = episode["fills"]
    f_ts  = np.array([f[0] for f in fills], dtype=np.int64)
    f_bid = np.array([f[2] if f[1] == "BID" else 0 for f in fills], dtype=np.int64)
    f_ask = np.array([f[2] if f[1] == "ASK" else 0 for f in fills], dtype=np.int64)
    cum_bid = np.concatenate([[0], np.cumsum(f_bid)])
    cum_ask = np.concatenate([[0], np.cumsum(f_ask)])

    def fills_upto(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = np.searchsorted(f_ts, t, side="right")
        return cum_bid[k], cum_ask[k]

    bid_to_t, ask_to_t = fills_upto(ts)
    bid_to_p, ask_to_p = fills_upto(prev)
    interval_bid = bid_to_t - np.where(prev < 0, 0, bid_to_p)
    interval_ask = ask_to_t - np.where(prev < 0, 0, ask_to_p)
    interval_bid[0] = interval_ask[0] = 0             # t0 row: nothing has happened yet

    m = np.searchsorted(marks, ts, side="right") - 1  # last decision at or before t
    base_inv = np.array([0] + [r["inventory"] for r in steps])[m]
    mb, ma = fills_upto(marks[m])
    inv = np.clip(base_inv + (bid_to_t - mb) - (ask_to_t - ma), -Q_max, Q_max)

    # ── Quotes in force: action of the decision that precedes/ends t ─────
    max_offset = float(TICK_OFFSETS[-1])
    has_quote  = ts > t0
    row = np.clip(np.searchsorted(marks[1:], ts, side="left"), 0, max(len(steps) - 1, 0))
    def step_col(name: str) -> np.ndarray:
        vals = np.array([r[name] for r in steps], dtype=float) if steps else np.zeros(1)
        return np.where(has_quote, vals[row], np.nan)
    bid_offset, ask_offset = step_col("bid_offset"), step_col("ask_offset")
    bid_price,  ask_price  = step_col("bid_price"),  step_col("ask_price")

    # ── Features that depend on history over grid points ─────────────────
    price_hist  = deque(maxlen=_PRICE_HISTORY_MAXLEN)
    volume_hist = deque(maxlen=_VOLUME_HISTORY_MAXLEN)
    mid_move = np.zeros(n); vol = np.zeros(n); rsi = np.zeros(n); sig_vol = np.zeros(n)
    for i in range(n):
        price_hist.append(mid[i])
        if i > 0:
            mid_move[i] = np.clip(np.log(mid[i] / mid[i - 1]) / tick_size, -10.0, 10.0)
            volume_hist.append(float(interval_bid[i] - interval_ask[i]))
            scale = max(abs(v) for v in volume_hist) + 1e-8
            sig_vol[i] = np.clip(volume_hist[-1] / scale, -1.0, 1.0)
        vol[i] = np.clip(compute_realized_vol(price_hist, vol_window) / max(tick_size, 1e-8), 0.0, 50.0)
        rsi[i] = (compute_rsi(price_hist, rsi_window) - 50.0) / 50.0

    # ── Market-wide order flow: latest trailing-window volume push ───────
    vol_events = episode.get("volume", [])
    vol_ts = np.array([v["ts"] for v in vol_events], dtype=np.int64)
    vidx = np.searchsorted(vol_ts, ts, side="right") - 1
    seen = vidx >= 0
    def vol_col(name: str) -> np.ndarray:
        vals = np.array([v[name] for v in vol_events], dtype=float)
        return np.where(seen, vals[np.maximum(vidx, 0)], np.nan) if len(vals) else np.full(n, np.nan)
    volume_age = (
        np.where(seen, (ts - vol_ts[np.maximum(vidx, 0)]) / _NS, np.nan)
        if len(vol_ts) else np.full(n, np.nan)
    )
    mkt_open = episode.get("mkt_open")

    # ── Assemble ──────────────────────────────────────────────────────────
    out = {
        "ts":        ts,
        "t_s":       (ts - t0) / _NS,
        "t_since_open_s": (ts - mkt_open) / _NS if mkt_open is not None else np.full(n, np.nan),
        "step":      m,
        "is_decision": np.isin(ts, marks),
        "mid":       mid,
        "best_bid":  ev_best_bid[idx],
        "best_ask":  ev_best_ask[idx],
        "book_age_s": (ts - ev_ts[idx]) / _NS,
        "n_book_updates": n_book,
        "mkt_bid_volume": vol_col("bid_volume"),
        "mkt_ask_volume": vol_col("ask_volume"),
        "volume_age_s":   volume_age,
        "bid_filled": interval_bid,
        "ask_filled": interval_ask,
        "inventory_shares": inv,
        "bid_offset": bid_offset,
        "ask_offset": ask_offset,
        "reward": np.full(n, np.nan),
    }
    for j, r in enumerate(steps):                      # reward lives on decision rows only
        hit = np.nonzero(ts == marks[j + 1])[0]
        if len(hit):
            out["reward"][hit[0]] = r["reward"]

    feats = {
        "spread": ev_spread[idx], "mid_move": mid_move, "imbalance": ev_imbal[idx],
        "signed_vol": sig_vol, "vol": vol, "rsi": rsi,
    }
    for i in range(n_lob_levels):
        feats[f"bid_depth_{i + 1}"] = ev_bid_d[idx, i]
    for i in range(n_lob_levels):
        feats[f"ask_depth_{i + 1}"] = ev_ask_d[idx, i]
    feats["inventory"] = np.clip(inv / Q_max, -1.0, 1.0)
    feats["bid_dist"]  = np.where(has_quote, np.clip(bid_offset / max_offset, 0.0, 1.0), 0.0)
    feats["ask_dist"]  = np.where(has_quote, np.clip(ask_offset / max_offset, 0.0, 1.0), 0.0)
    span = max_offset * tick_size
    feats["outstanding_bid"] = np.where(has_quote, np.clip((bid_price - mid) / span, -1.0, 1.0), 0.0)
    feats["outstanding_ask"] = np.where(has_quote, np.clip((ask_price - mid) / span, -1.0, 1.0), 0.0)
    feats["time_remaining"] = (
        1.0 - (ts - t0) / interval / episode_len if interval else np.ones(n)
    )
    out.update(feats)
    return pd.DataFrame(out)
