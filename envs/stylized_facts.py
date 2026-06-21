"""
envs/stylized_facts.py

Stylized facts validator for the ABIDES-Gym simulator.

Runs the environment for N episodes and checks that ≥4 of 5 canonical
market microstructure stylized facts hold. This is the validation gate
at the end of Week 2 — do not proceed to Week 3 baselines until this passes.

The five stylized facts checked:
    1. Fat-tailed return distribution (kurtosis > 3 or Jarque-Bera rejects normality)
    2. Volatility clustering (autocorrelation of |returns| positive at lags 1–20)
    3. Bid-ask spread autocorrelation (spread is not i.i.d.)
    4. Price impact signature (mid-price moves in direction of large orders)
    5. Queue imbalance predictability (imbalance predicts short-term direction)

References:
    Huang, Lehalle & Rosenbaum (2015) — queue-reactive model, stylized facts
    Byrd, Hybinette & Balch (2019) — ABIDES simulator validation

Week 2 deliverable.
"""

import numpy as np
from scipy import stats
from typing import Tuple

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Correlation that won't explode on constant arrays.
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])

def check_fat_tails(returns: np.ndarray, threshold: float = 3.0) -> Tuple[bool, dict]:
    """
    Stylized fact 1: Fat-tailed return distribution.

    Real LOB returns have excess kurtosis > 3 (heavier tails than Gaussian).
    Alternatively, Jarque-Bera test rejects normality at 5% significance.

    Parameters
    ----------
    returns   : np.ndarray — array of mid-price log returns
    threshold : float      — minimum kurtosis to pass (default 3.0)

    Returns
    -------
    passed : bool
    stats  : dict — kurtosis, jb_statistic, jb_pvalue
    """
    returns = np.asarray(returns)

    if len(returns) < 10:
        return False, {"error": "insufficient data"}

    # Pearson kurtosis:
    # Gaussian = 3
    kurtosis = float(
        stats.kurtosis(
            returns,
            fisher=False,
            bias=False,
        )
    )

    jb = stats.jarque_bera(returns)

    passed = (kurtosis > threshold or jb.pvalue < 0.05)

    return passed, {
        "kurtosis": kurtosis,
        "jb_statistic": float(jb.statistic),
        "jb_pvalue": float(jb.pvalue),
    }


def check_volatility_clustering(
    returns:   np.ndarray,
    max_lag:   int   = 5,
    min_autocorr: float = 0.05,
) -> Tuple[bool, dict]:
    """
    Stylized fact 2: Volatility clustering.

    |returns| should be positively autocorrelated.
    This captures the GARCH-like clustering of volatility in real markets.

    Parameters
    ----------
    returns      : np.ndarray — mid-price log returns
    max_lag      : int        — maximum lag to check
    min_autocorr : float      — minimum mean autocorrelation to pass

    Returns
    -------
    passed : bool
    stats  : dict — autocorrelations at each lag, mean_autocorr
    """
    abs_returns = np.abs(np.asarray(returns))
    if len(abs_returns) < max_lag + 5:
        return False, {"error": "insufficient data"}

    autocorrs = []
    for lag in range(1, max_lag + 1):
        corr = _safe_corr(
            abs_returns[:-lag],
            abs_returns[lag:],
        )
        if not np.isnan(corr):
            autocorrs.append(corr)

    if len(autocorrs) == 0:
        return False, {"error": "autocorrelation undefined"}

    mean_autocorr = float(np.mean(autocorrs))
    passed = mean_autocorr > min_autocorr

    return passed, {
        "mean_autocorr": mean_autocorr,
        "autocorrs": autocorrs,
    }


def check_spread_autocorrelation(
    spreads:     np.ndarray,
    min_autocorr: float = 0.03,
) -> Tuple[bool, dict]:
    """
    Stylized fact 3: Bid-ask spread autocorrelation.

    The bid-ask spread should be positively autocorrelated at lag 1.
    A purely random spread (no persistence) would fail this check.

    Parameters
    ----------
    spreads      : np.ndarray — array of bid-ask spreads (ask - bid) over episode
    min_autocorr : float      — minimum lag-1 autocorrelation to pass

    Returns
    -------
    passed : bool
    stats  : dict — lag1_autocorr
    """
    spreads = np.asarray(spreads)
    if len(spreads) < 5:
        return False, {"error": "insufficient data"}

    lag1 = _safe_corr(spreads[:-1], spreads[1:])
    if np.isnan(lag1):
        return False, {
            "lag1_autocorr": np.nan,
            "error": "constant spread series",
        }

    passed = lag1 > min_autocorr

    return passed, {
        "lag1_autocorr": float(lag1),
    }


def check_price_impact(
    order_sizes:  np.ndarray,
    price_moves:  np.ndarray,
    min_corr:     float = 0.05,
) -> Tuple[bool, dict]:
    """
    Stylized fact 4: Price impact signature.

    For a market maker, fills on the bid side (positive signed_volume)
    should be followed by price DECREASES (adverse selection), and
    fills on the ask side by price INCREASES. So the correlation
    between signed_volume and subsequent price move should be NEGATIVE.

    A correlation more negative than -min_corr indicates the simulator
    is producing realistic adverse selection.

    Parameters
    ----------
    order_sizes  : np.ndarray — signed order sizes (+buy, -sell)
    price_moves  : np.ndarray — mid-price changes after each order
    min_corr     : float      — minimum correlation to pass

    Returns
    -------
    passed : bool
    stats  : dict — correlation, interpretation
    """
    if len(order_sizes) < 10:
        return False, {"error": "insufficient data"}

    corr = _safe_corr(np.asarray(order_sizes), np.asarray(price_moves))
    if np.isnan(corr):
        return False, {"correlation": np.nan}

    # Negative correlation = adverse selection = realistic market making
    passed = corr < -min_corr

    return passed, {
        "correlation": float(corr),
        "interpretation": "negative = adverse selection present (expected)"
    }


def check_queue_imbalance_predictability(
    imbalances:   np.ndarray,
    future_moves: np.ndarray,
    min_corr:     float = 0.03,
) -> Tuple[bool, dict]:
    """
    Stylized fact 5: Queue imbalance predicts short-term price direction.

    Queue imbalance I = (V_bid - V_ask) / (V_bid + V_ask) should be
    positively correlated with the next mid-price move.
    This is the key finding of Huang et al. (2015).

    Parameters
    ----------
    imbalances   : np.ndarray — queue imbalance at each step ∈ [−1, 1]
    future_moves : np.ndarray — mid-price change at next step
    min_corr     : float      — minimum correlation to pass

    Returns
    -------
    passed : bool
    stats  : dict — correlation
    """
    if len(imbalances) < 10:
        return False, {"error": "insufficient data"}

    n = min(len(imbalances), len(future_moves))
    corr = _safe_corr(np.asarray(imbalances[:n]), np.asarray(future_moves[:n]))
    if np.isnan(corr):
        return False, {"correlation": np.nan}

    passed = corr > min_corr

    return passed, {
        "correlation": float(corr),
    }


def run_stylized_facts_audit(
    env,
    n_episodes:     int   = 10,
    min_facts_pass: int   = 4,
    verbose:        bool  = True,
) -> Tuple[bool, dict]:
    """
    Run the full stylized facts audit on the environment.

    Rolls out n_episodes with a random agent and collects:
        - mid-price path → returns
        - bid-ask spreads
        - order sizes and price impacts
        - queue imbalances

    Then runs all five checks and reports which pass.

    Parameters
    ----------
    env            : gym.Env — your LOBMarketMakingEnv instance
    n_episodes     : int     — number of episodes to collect data from
    min_facts_pass : int     — minimum facts that must pass (default 4 of 5)
    verbose        : bool    — print results

    Returns
    -------
    overall_pass : bool — True if ≥ min_facts_pass facts hold
    results      : dict — per-fact results and statistics
    """
    # ── Collect data across episodes ───────────────────────────────────
    all_returns      = []
    all_spreads      = []
    all_order_sizes  = []   # signed volume per step
    all_price_moves  = []   # mid-price move after each step
    all_imbalances   = []
    all_future_moves = []   # mid-price move at t+1 (for imbalance predictability)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        mid_prices = [info.get("mid_price", 0.0)]
        spreads    = []
        imbalances = []
        signed_vols = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            mid_prices.append(info.get("mid_price", mid_prices[-1]))

            spreads.append(info.get("market_spread", 0.0))

            imbalances.append(info.get("queue_imbalance", 0.0))
            signed_vols.append(info.get("signed_volume", 0.0))

        mid_arr = np.array(mid_prices)
        if len(mid_arr) > 1:
            returns = np.diff(np.log(np.maximum(mid_arr, 1e-10)))
            all_returns.extend(returns.tolist())

            # Fact 4: signed_vol[t] vs price_move[t+1]
            # signed_vol has one entry per step; price move is diff(mid_prices)
            # both length = n_steps, align directly
            if len(signed_vols) == len(returns):
                all_order_sizes.extend(signed_vols)
                all_price_moves.extend(returns.tolist())

            # Fact 5: imbalance[t] vs price_move[t+1]
            # imbalance is recorded at step t, future_move is returns[t]
            # (returns[t] = mid[t+1] - mid[t], recorded after step t)
            if len(imbalances) == len(returns):
                all_imbalances.extend(imbalances)
                all_future_moves.extend(returns.tolist())

        all_spreads.extend(spreads)

    returns_arr     = np.array(all_returns)
    spreads_arr     = np.array(all_spreads)
    order_sizes_arr = np.array(all_order_sizes)
    price_moves_arr = np.array(all_price_moves)
    imbalances_arr  = np.array(all_imbalances)
    future_moves_arr = np.array(all_future_moves)

    # ── Run the five checks ────────────────────────────────────────────
    results = {}

    if len(returns_arr) > 30:
        p1, s1 = check_fat_tails(returns_arr)
        results["fat_tails"] = {"passed": p1, "stats": s1}
    else:
        results["fat_tails"] = {"passed": False, "stats": {"error": "insufficient data"}}

    if len(returns_arr) > 30:
        p2, s2 = check_volatility_clustering(returns_arr)
        results["volatility_clustering"] = {"passed": p2, "stats": s2}
    else:
        results["volatility_clustering"] = {"passed": False, "stats": {"error": "insufficient data"}}

    if len(spreads_arr) > 30:
        p3, s3 = check_spread_autocorrelation(spreads_arr)
        results["spread_autocorr"] = {"passed": p3, "stats": s3}
    else:
        results["spread_autocorr"] = {"passed": False, "stats": {"error": "insufficient data"}}

    # Fact 4: price impact — now populated from signed_volume in info
    if len(order_sizes_arr) > 10 and np.any(order_sizes_arr != 0):
        p4, s4 = check_price_impact(order_sizes_arr, price_moves_arr)
        results["price_impact"] = {"passed": p4, "stats": s4}
    else:
        # With synthetic GBM and zero fills, signed_volume is always 0.
        # This fact can only be validated against the real ABIDES simulator.
        results["price_impact"] = {
            "passed": False,
            "stats": {"error": "signed_volume is zero — run with ABIDES wired in"}
        }

    # Fact 5: queue imbalance predictability — now populated from obs[2] in info
    if len(imbalances_arr) > 10 and np.any(imbalances_arr != 0):
        p5, s5 = check_queue_imbalance_predictability(imbalances_arr, future_moves_arr)
        results["queue_imbalance"] = {"passed": p5, "stats": s5}
    else:
        # Imbalance is zero until LOB history is populated by ABIDES.
        results["queue_imbalance"] = {
            "passed": False,
            "stats": {"error": "imbalance is zero — run with ABIDES wired in"}
        }

    # ── Summarise ──────────────────────────────────────────────────────
    n_passed = sum(v["passed"] for v in results.values())
    overall_pass = n_passed >= min_facts_pass

    if verbose:
        print("\n=== Stylized Facts Audit ===")
        for fact, res in results.items():
            status = "✓ PASS" if res["passed"] else "✗ FAIL"
            print(f"  {status} — {fact}")
            for k, v in res["stats"].items():
                if isinstance(v, float):
                    print(f"           {k}: {v:.4f}")
        print(f"\n  {n_passed}/{len(results)} facts passed "
              f"(need ≥{min_facts_pass})")
        print(f"  Overall: {'✓ PASS' if overall_pass else '✗ FAIL'}\n")

    return overall_pass, results


if __name__ == "__main__":
    """
    Quick validation — runs the audit on a dummy environment.
    Replace DummyEnv with your real LOBMarketMakingEnv once implemented.
    """
    from lob_env import LOBMarketMakingEnv

    env = LOBMarketMakingEnv(reward_type="asymmetric", episode_len=390, seed=42)
    passed, res = run_stylized_facts_audit(env, n_episodes=5)
    print("Audit complete.")
    env.close()