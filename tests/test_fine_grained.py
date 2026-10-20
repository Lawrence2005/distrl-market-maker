"""
tests/test_fine_grained.py
--------------------------
Fine-grained market-data recording: MarketDataConfig, the shared book-feature
helpers, and envs/fine_grained.py's per-second feature frame.

The first two classes need no simulator. The ABIDES class runs the real
simulator (slow, abides_gym required) and is marked so it can be excluded:
    pytest tests/ -v -m "not abides"
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from envs.fine_grained import build_grid_frame, feature_columns, merge_book_volume
from envs.lob_env import (
    LOBMarketMakingEnv,
    MarketDataConfig,
    book_features,
    format_book_events,
    mid_from_book,
)

NS = 1_000_000_000


# ── MarketDataConfig / shared helpers ────────────────────────────────────────

class TestMarketDataConfig:
    def test_defaults_are_abides_gym_defaults(self):
        cfg = MarketDataConfig()
        assert cfg.buffer_length == 5
        assert cfg.subscribe_freq_ns == 100_000_000
        assert cfg.timestep_duration is None
        assert cfg.record is False

    @pytest.mark.parametrize("kwargs", [{"buffer_length": 0}, {"subscribe_freq_ns": 0}])
    def test_rejects_nonpositive(self, kwargs):
        with pytest.raises(ValueError):
            MarketDataConfig(**kwargs)

    def test_env_without_abides_has_no_extra_info(self):
        env = LOBMarketMakingEnv(use_abides=False, episode_len=3, seed=0)
        _, info = env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([1, 1]))
        assert "book_events" not in info and "fill_events" not in info


class TestSharedBookHelpers:
    def test_mid_from_book_cents_to_dollars(self):
        assert mid_from_book([(99900, 5)], [(100100, 7)], 0) == pytest.approx(1000.0)

    def test_mid_falls_back_to_last_transaction_on_empty_side(self):
        assert mid_from_book([], [(100100, 7)], 100000) == pytest.approx(1000.5)

    def test_book_features_zero_without_snapshot(self):
        spread, imb, bid_d, ask_d = book_features(None, 0.01, 3)
        assert spread == 0.0 and imb == 0.0
        assert bid_d.shape == ask_d.shape == (3,) and not bid_d.any() and not ask_d.any()

    def test_book_features_match_env_observation(self):
        """The env's obs entries [0], [2], [6:6+2K] are exactly book_features."""
        env = LOBMarketMakingEnv(use_abides=False, n_lob_levels=3, seed=0)
        env.reset(seed=0)
        snap = {
            "bid_prices": [99.99, 99.98, 99.97, 99.96], "bid_sizes": [300, 200, 100, 50],
            "ask_prices": [100.02, 100.03, 100.05],     "ask_sizes": [100, 400, 250],
        }
        env._lob_history.append(snap)
        obs = env._get_obs()
        spread, imb, bid_d, ask_d = book_features(snap, env.tick_size, 3)
        assert obs[0] == np.float32(spread)
        assert obs[2] == np.float32(imb)
        np.testing.assert_array_equal(obs[6:9], bid_d)
        np.testing.assert_array_equal(obs[9:12], ask_d)

    def test_format_book_events_dollars_and_depth_cap(self):
        raw = [{"exchange_ts": 5, "last_transaction": 100000,
                "bids": [(99900, 5), (99800, 6)], "asks": [(100100, 7)]}]
        (e,) = format_book_events(raw, n_levels=1)
        assert e["ts"] == 5 and e["mid"] == pytest.approx(1000.0)
        assert e["bid_prices"] == [999.0] and e["bid_sizes"] == [5]
        assert e["ask_prices"] == [1001.0] and e["last_transaction"] == pytest.approx(1000.0)


# ── build_grid_frame on a hand-built episode ─────────────────────────────────

T0 = 1_000 * NS


def _snap(t_s, bid, ask, bid_size=100, ask_size=100):
    return {"ts": T0 + int(t_s * NS), "mid": (bid + ask) / 2, "last_transaction": bid,
            "bid_prices": [bid, bid - 0.01], "bid_sizes": [bid_size, 50],
            "ask_prices": [ask, ask + 0.01], "ask_sizes": [ask_size, 50]}


def _episode():
    """Two 60 s decisions; one BID fill of 100 sh 30 s in; one ASK fill at 90 s."""
    steps = [
        dict(step=1, decision_ts=T0 + 60 * NS, bid_offset=10, ask_offset=20,
             bid_price=99.90, ask_price=100.20, reward=1.5, inventory=100),
        dict(step=2, decision_ts=T0 + 120 * NS, bid_offset=30, ask_offset=40,
             bid_price=99.70, ask_price=100.40, reward=-0.5, inventory=0),
    ]
    return {
        "t0": T0,
        "book": [_snap(-5, 99.99, 100.01), _snap(10, 100.00, 100.02, bid_size=300),
                 _snap(70, 100.05, 100.07)],
        "fills": [(T0 + 30 * NS, "BID", 100, 99900), (T0 + 90 * NS, "ASK", 100, 100200)],
        "volume": [{"ts": T0 + 25 * NS, "bid_volume": 300, "ask_volume": 500},
                   {"ts": T0 + 85 * NS, "bid_volume": 450, "ask_volume": 100}],
        "mkt_open": T0 - 30 * NS,
        "steps": steps,
    }


@pytest.fixture(scope="module")
def grid():
    return build_grid_frame(
        _episode(), grid_s=10.0, tick_size=0.01, n_lob_levels=2,
        Q_max=1000, episode_len=4,
    ).set_index("t_s")


class TestMergeBookVolume:
    def test_volume_columns_are_appended_row_for_row(self):
        book = pd.DataFrame({"ts": [1, 2, 3], "mid": [10.0, 11.0, 12.0]})
        vol = pd.DataFrame({"ts": [1, 2, 3], "bid_volume": [5, 6, 7], "ask_volume": [8, 9, 10]})
        out = merge_book_volume(book, vol)
        assert list(out.columns) == ["ts", "mid", "mkt_bid_volume", "mkt_ask_volume"]
        assert out.mkt_bid_volume.tolist() == [5, 6, 7] and out.mkt_ask_volume.tolist() == [8, 9, 10]

    def test_unpaired_events_raise(self):
        book = pd.DataFrame({"ts": [1, 2, 3]})
        with pytest.raises(ValueError):
            merge_book_volume(book, pd.DataFrame({"ts": [1, 2], "bid_volume": [1, 1], "ask_volume": [1, 1]}))
        with pytest.raises(ValueError):
            merge_book_volume(book, pd.DataFrame({"ts": [1, 2, 4], "bid_volume": [1] * 3, "ask_volume": [1] * 3}))


class TestBuildGridFrame:
    def test_shape_and_feature_columns(self, grid):
        assert list(grid.index) == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                                    70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
        assert set(feature_columns(2)) <= set(grid.columns)
        assert len(feature_columns(3)) == 18       # the handcrafted observation

    def test_decision_flag_and_reward_only_on_decision_rows(self, grid):
        assert list(grid.index[grid.is_decision]) == [0.0, 60.0, 120.0]
        assert grid.reward.dropna().to_dict() == {60.0: 1.5, 120.0: -0.5}

    def test_book_is_forward_filled_with_age(self, grid):
        assert grid.loc[0.0, "book_age_s"] == pytest.approx(5.0)      # snapshot at -5 s
        assert grid.loc[10.0, "book_age_s"] == pytest.approx(0.0)
        assert grid.loc[60.0, "mid"] == pytest.approx(100.01)         # still the 10 s snapshot
        assert grid.loc[70.0, "mid"] == pytest.approx(100.06)

    def test_inventory_follows_timestamped_fills(self, grid):
        assert grid.loc[20.0, "inventory_shares"] == 0
        assert grid.loc[30.0, "inventory_shares"] == 100              # fill at exactly 30 s
        assert grid.loc[60.0, "inventory_shares"] == 100              # env's post-step value
        assert grid.loc[80.0, "inventory_shares"] == 100
        assert grid.loc[90.0, "inventory_shares"] == 0                # sold at 90 s
        assert grid.loc[30.0, "bid_filled"] == 100 and grid.loc[90.0, "ask_filled"] == 100

    def test_quotes_are_held_until_next_decision(self, grid):
        assert np.isnan(grid.loc[0.0, "bid_offset"])                  # nothing quoted at reset
        assert grid.loc[10.0, "bid_offset"] == 10 and grid.loc[60.0, "bid_offset"] == 10
        assert grid.loc[70.0, "bid_offset"] == 30 and grid.loc[120.0, "ask_offset"] == 40
        assert grid.loc[60.0, "bid_dist"] == pytest.approx(10 / 100)

    def test_time_remaining_counts_down_by_decisions(self, grid):
        assert grid.loc[0.0, "time_remaining"] == pytest.approx(1.0)
        assert grid.loc[60.0, "time_remaining"] == pytest.approx(1 - 1 / 4)
        assert grid.loc[30.0, "time_remaining"] == pytest.approx(1 - 0.5 / 4)

    def test_state_columns_are_set_on_every_row(self, grid):
        # inventory and the quote in force exist at every moment; only the
        # reset row has no quote yet (offsets NaN, distances 0 like the env's obs)
        for name in ("inventory", "inventory_shares", "bid_dist", "ask_dist",
                     "outstanding_bid", "outstanding_ask", "time_remaining",
                     "spread", "imbalance", "signed_vol", "bid_filled", "ask_filled"):
            assert grid[name].notna().all(), name
        assert grid.loc[10:, ["bid_offset", "ask_offset"]].notna().all().all()
        assert grid.loc[0.0, "bid_dist"] == 0.0 and np.isnan(grid.loc[0.0, "bid_offset"])

    def test_market_volume_is_forward_filled_with_age(self, grid):
        assert np.isnan(grid.loc[20.0, "mkt_bid_volume"])             # before first push
        assert grid.loc[30.0, "mkt_bid_volume"] == 300 and grid.loc[30.0, "mkt_ask_volume"] == 500
        assert grid.loc[30.0, "volume_age_s"] == pytest.approx(5.0)
        assert grid.loc[100.0, "mkt_bid_volume"] == 450               # latest push wins

    def test_time_since_open_uses_market_open(self, grid):
        assert grid.loc[0.0, "t_since_open_s"] == pytest.approx(30.0)  # first_interval offset
        assert grid.loc[60.0, "t_since_open_s"] == pytest.approx(90.0)

    def test_missing_volume_and_open_are_nan_not_errors(self):
        ep = _episode(); ep.pop("volume"); ep.pop("mkt_open")
        g = build_grid_frame(ep, grid_s=30.0, tick_size=0.01, n_lob_levels=2,
                             Q_max=1000, episode_len=4)
        assert g.mkt_bid_volume.isna().all() and g.t_since_open_s.isna().all()

    def test_grid_spacing_is_an_analysis_time_choice(self):
        fine = build_grid_frame(_episode(), grid_s=1.0, tick_size=0.01,
                                n_lob_levels=2, Q_max=1000, episode_len=4)
        coarse = build_grid_frame(_episode(), grid_s=60.0, tick_size=0.01,
                                  n_lob_levels=2, Q_max=1000, episode_len=4)
        assert len(fine) == 121 and len(coarse) == 3
        cols = feature_columns(2)
        # Rows at decision times agree except the history-windowed features,
        # whose windows count grid points (see build_grid_frame docstring).
        same = [c for c in cols if c not in ("mid_move", "signed_vol", "vol", "rsi")]
        pd_fine = fine[fine.is_decision][same].reset_index(drop=True)
        np.testing.assert_allclose(pd_fine.to_numpy(), coarse[same].to_numpy())

    def test_rejects_nonpositive_grid(self):
        with pytest.raises(ValueError):
            build_grid_frame(_episode(), grid_s=0, tick_size=0.01, n_lob_levels=2,
                             Q_max=1000, episode_len=4)


# ── Real ABIDES ──────────────────────────────────────────────────────────────

ABIDES_AVAILABLE = importlib.util.find_spec("abides_gym") is not None


def _rollout(market_data, n_steps, seed=7):
    """Seeded random-action rollout; returns stacked obs, rewards, last info."""
    from envs.fine_grained import FineGrainedRecorder
    env = LOBMarketMakingEnv(episode_len=390, seed=seed, market_data=market_data)
    rec = FineGrainedRecorder(env) if market_data and market_data.record else None
    drv = rec or env
    rng = np.random.default_rng(seed)
    obs, info = drv.reset(seed=seed)
    O, R = [obs], []
    for _ in range(n_steps):
        obs, r, te, tr, info = drv.step(np.array([rng.integers(11), rng.integers(11)]))
        O.append(obs); R.append(r)
        if te or tr:
            break
    return np.array(O), np.array(R), info, rec, env


@pytest.mark.abides
@pytest.mark.skipif(not ABIDES_AVAILABLE, reason="abides_gym not installed")
class TestRecordingWithAbides:
    def test_default_info_has_no_extra_keys(self):
        _, _, info, _, env = _rollout(None, 2)
        env.close()
        assert not {"book_events", "fill_events", "decision_ts"} & set(info)

    def test_recording_leaves_trajectory_unchanged(self):
        """The central claim: record=True must not perturb the simulation."""
        o0, r0, _, _, e0 = _rollout(None, 25)
        o1, r1, _, _, e1 = _rollout(MarketDataConfig(record=True), 25)
        e0.close(); e1.close()
        np.testing.assert_array_equal(o0, o1)
        np.testing.assert_array_equal(r0, r1)

    def test_grid_rows_at_decisions_equal_agent_observations(self):
        obs, _, _, rec, env = _rollout(MarketDataConfig(record=True), 30)
        frames = rec.frames(grid_s=60.0)
        env.close()
        g = frames["grid"]
        got = g[g.is_decision][feature_columns(env.n_lob_levels)].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(got, obs, atol=1e-5)

    def test_events_are_complete_ordered_and_inside_the_window(self):
        _, _, _, rec, env = _rollout(MarketDataConfig(record=True), 20)
        fr = rec.frames(grid_s=1.0)
        env.close()
        assert len(fr["volume"]) > 0 and (fr["volume"].ts.diff().dropna() >= 0).all()
        assert fr["grid"].mkt_bid_volume.notna().any()
        merged = merge_book_volume(fr["book"], fr["volume"])    # real streams are paired
        assert len(merged) == len(fr["book"])
        assert (fr["book"].ts.diff().dropna() >= 0).all()
        assert len(fr["book"]) > 20                    # many more than one per decision
        assert fr["fills"].ts.between(fr["steps"].decision_ts.min() - 60 * NS,
                                      fr["steps"].decision_ts.max()).all()
        assert fr["steps"].decision_ts.diff().dropna().eq(60 * NS).all()

    def test_full_episode_reaches_market_close(self):
        """A gym-agent subclass the kernel doesn't recognise fails exactly here
        (step 390 returns no final state) — see RecordingGymAgent's docstring."""
        env = LOBMarketMakingEnv(episode_len=390, seed=3,
                                 market_data=MarketDataConfig(record=True))
        env.reset(seed=3)
        steps, truncated = 0, False
        while not truncated:
            _, _, _, truncated, _ = env.step(np.array([3, 3]))
            steps += 1
        env.close()
        assert steps == 390

    def test_recorder_requires_recording_env(self):
        from envs.fine_grained import FineGrainedRecorder
        env = LOBMarketMakingEnv(episode_len=5, seed=0)
        with pytest.raises(ValueError):
            FineGrainedRecorder(env)
        env.close()
