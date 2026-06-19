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
    max_lag:   int   = 20,
    min_autocorr: float = 0.05,
) -> Tuple[bool, dict]:
    """
    Stylized fact 2: Volatility clustering.

    |returns| should be positively autocorrelated at lags 1–20.
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
    min_autocorr: float = 0.1,
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

    Mid-price should move in the direction of large orders.
    Correlation between signed order size and subsequent price move
    should be positive.

    Parameters
    ----------
    order_sizes  : np.ndarray — signed order sizes (+buy, -sell)
    price_moves  : np.ndarray — mid-price changes after each order
    min_corr     : float      — minimum correlation to pass

    Returns
    -------
    passed : bool
    stats  : dict — correlation
    """
    if len(order_sizes) < 10:
        return False, {"error": "insufficient data"}

    corr = _safe_corr(np.asarray(order_sizes), np.asarray(price_moves))
    if np.isnan(corr):
        return False, {"correlation": np.nan}

    passed = corr > min_corr

    return passed, {
        "correlation": float(corr),
    }


def check_queue_imbalance_predictability(
    imbalances:   np.ndarray,
    future_moves: np.ndarray,
    min_corr:     float = 0.05,
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
    all_returns     = []
    all_spreads     = []
    all_order_sizes = []
    all_price_moves = []
    all_imbalances  = []
    all_future_moves = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        mid_prices = [info.get("mid_price", 0.0)]
        spreads    = []
        imbalances = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()   # random agent
            obs, reward, terminated, truncated, info = env.step(action)

            mid_prices.append(
                info.get("mid_price", mid_prices[-1])
            )

            spreads.append(
                info.get("ask_price", 0.0) - info.get("bid_price", 0.0)
            )

            imbalances.append(
                info.get("queue_imbalance", 0.0)
            )

        # Compute returns from mid-price path
        mid_arr = np.array(mid_prices)
        if len(mid_arr) > 1:
            returns = np.diff(np.log(np.maximum(mid_arr, 1e-10)))
            all_returns.extend(returns.tolist())

        all_spreads.extend(spreads)
        all_imbalances.extend(imbalances)

    # Convert to arrays
    returns_arr    = np.array(all_returns)
    spreads_arr    = np.array(all_spreads)
    imbalances_arr = np.array(all_imbalances)

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

    # Facts 4 and 5 require order size and imbalance data from info dict
    # TODO: populate once lob_env.py provides these in info
    results["price_impact"]   = {"passed": False, "stats": {"error": "TODO: needs order size data from env"}}
    results["queue_imbalance"] = {"passed": False, "stats": {"error": "TODO: needs imbalance data from env"}}

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
    import gymnasium as gym

    class DummyEnv(gym.Env):
        """Placeholder until lob_env.py is implemented."""
        def __init__(self):
            self.action_space      = gym.spaces.Discrete(81)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
            )

        def reset(self, seed=None, options=None):
            return np.zeros(17, dtype=np.float32), {"mid_price": 150.0}

        def step(self, action):
            obs    = np.random.randn(17).astype(np.float32)
            reward = np.random.randn()
            done   = np.random.rand() < 0.005
            return obs, reward, done, False, {"mid_price": 150.0 + np.random.randn()}

    env          = DummyEnv()
    passed, res  = run_stylized_facts_audit(env, n_episodes=5)
    print("Audit complete. Replace DummyEnv with LOBMarketMakingEnv.")
