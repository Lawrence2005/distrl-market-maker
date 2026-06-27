# import inspect
# import textwrap
# import numpy as np

# from abides_gym.envs import SubGymMarketsDailyInvestorEnv_v0


# def banner(msg):
#     print("\n" + "=" * 80)
#     print(msg)
#     print("=" * 80)


# def print_source(obj, name):
#     try:
#         banner(f"SOURCE: {name}")
#         print(inspect.getsource(obj))
#     except Exception as e:
#         print(f"Could not get source for {name}: {e}")


# def safe_reset(env):
#     try:
#         result = env.reset()

#         if isinstance(result, tuple):
#             print("Gymnasium-style reset detected")
#             print("Tuple length:", len(result))
#             return result

#         print("Old Gym-style reset detected")
#         return result

#     except Exception as e:
#         print("Reset failed:", e)
#         return None


# def safe_step(env):
#     try:
#         action = env.action_space.sample()

#         banner("SAMPLED ACTION")
#         print(action)

#         result = env.step(action)

#         banner("STEP RESULT")
#         print("type:", type(result))
#         print("len :", len(result))

#         for i, item in enumerate(result):
#             print(f"\nITEM {i}")
#             print("TYPE:", type(item))
#             print(item)

#         return result

#     except Exception as e:
#         print("Step failed:", e)
#         return None


# def inspect_env_class(env):

#     cls = type(env)

#     banner("ENVIRONMENT CLASS")
#     print(cls)
#     print("module:", cls.__module__)
#     print("file  :", inspect.getfile(cls))

#     banner("SPACES")
#     print("Observation space:")
#     print(env.observation_space)

#     print("\nAction space:")
#     print(env.action_space)

#     banner("INSTANCE ATTRIBUTES")
#     for k, v in env.__dict__.items():
#         print(f"{k:35s} {type(v)}")

#     banner("METHODS CONTAINING KEYWORDS")

#     keywords = [
#         "state",
#         "action",
#         "raw",
#         "reward",
#         "market",
#         "book",
#         "lob",
#         "obs",
#         "step",
#         "reset",
#     ]

#     methods = []

#     for name in dir(cls):
#         lname = name.lower()

#         if any(k in lname for k in keywords):
#             methods.append(name)

#     for m in sorted(methods):
#         print(m)

#     return cls


# def inspect_gym_agent(env):

#     banner("GYM AGENT CONSTRUCTOR")

#     agent_cls = getattr(env, "gymAgentConstructor", None)

#     if agent_cls is None:
#         print("No gymAgentConstructor found")
#         return

#     print(agent_cls)

#     try:
#         print("module:", agent_cls.__module__)
#         print("file:", inspect.getfile(agent_cls))
#     except Exception as e:
#         print(e)

#     try:
#         banner("GYM AGENT METHODS")

#         for name in dir(agent_cls):
#             lname = name.lower()

#             if any(
#                 k in lname
#                 for k in [
#                     "state",
#                     "action",
#                     "market",
#                     "book",
#                     "lob",
#                     "obs",
#                     "reward",
#                 ]
#             ):
#                 print(name)

#     except Exception as e:
#         print(e)

#     # Try common schema methods
#     candidate_methods = [
#         "raw_state_to_state",
#         "raw_state_to_reward",
#         "raw_state_to_info",
#         "map_action_space_to_ABIDES_SIMULATOR_SPACE",
#         "kernel_starting",
#         "wakeup",
#         "act_on_wakeup",
#     ]

#     for method_name in candidate_methods:

#         if hasattr(agent_cls, method_name):

#             try:
#                 method = getattr(agent_cls, method_name)

#                 banner(f"AGENT SOURCE: {method_name}")
#                 print(inspect.getsource(method))

#             except Exception as e:
#                 print(method_name, e)


# def main():

#     banner("CREATE ENV")

#     env = SubGymMarketsDailyInvestorEnv_v0()

#     cls = inspect_env_class(env)

#     banner("RESET")

#     obs = safe_reset(env)

#     banner("OBSERVATION DETAILS")

#     if isinstance(obs, tuple):
#         print("reset returned tuple")
#         print(obs)

#     else:
#         print("type :", type(obs))

#         if isinstance(obs, np.ndarray):
#             print("shape:", obs.shape)
#             print(obs)

#     safe_step(env)

#     banner("CLASS SOURCE")

#     for name in [
#         "__init__",
#         "reset",
#         "step",
#     ]:
#         if hasattr(cls, name):
#             print_source(getattr(cls, name), name)

#     banner("SEARCH FOR STATE/ACTION METHODS")

#     for name in dir(cls):

#         lname = name.lower()

#         if (
#             "state" in lname
#             or "action" in lname
#             or "reward" in lname
#             or "obs" in lname
#         ):
#             attr = getattr(cls, name)

#             if callable(attr):
#                 print_source(attr, name)

#     inspect_gym_agent(env)

#     banner("DONE")


# if __name__ == "__main__":
#     main()
# --------------------------------------------------------------------------------
# import inspect
# from abides_gym.experimental_agents.financial_gym_agent import FinancialGymAgent

# targets = [
#     "get_internal_data",
#     "get_parsed_mkt_data",
#     "get_parsed_volume_data",
# ]

# for t in targets:
#     print("\n" + "=" * 80)
#     print(t)
#     print("=" * 80)
#     print(inspect.getsource(getattr(FinancialGymAgent, t)))

import inspect
import abides_gym
import abides_gym.envs as envs

# ── 1. What environments are available ───────────────────────────────
print("=" * 60)
print("AVAILABLE ENV CLASSES")
print("=" * 60)
import inspect, pkgutil
for name, obj in inspect.getmembers(envs, inspect.isclass):
    print(f"  {name}")

# ── 2. Pick the base environment class and print its source ──────────
from abides_gym.envs.markets_environment import AbidesGymMarketsEnv
print("\n" + "=" * 60)
print("AbidesGymMarketsEnv SOURCE")
print("=" * 60)
print(inspect.getsource(AbidesGymMarketsEnv))

# ── 3. Print the daily investor env (closest to market making) ───────
from abides_gym.envs.markets_execution_environment_v0 import SubGymMarketsExecutionEnv_v0
print("\n" + "=" * 60)
print("SubGymMarketsExecutionEnv_v0 SOURCE")
print("=" * 60)
print(inspect.getsource(SubGymMarketsExecutionEnv_v0))

# ── 4. Observation space layout ──────────────────────────────────────
env = SubGymMarketsExecutionEnv_v0()
print("\n" + "=" * 60)
print("OBSERVATION SPACE")
print("=" * 60)
print(env.observation_space)

# ── 5. Action space layout ───────────────────────────────────────────
print("\n" + "=" * 60)
print("ACTION SPACE")
print("=" * 60)
print(env.action_space)

# ── 6. Do a reset and inspect raw obs + info ─────────────────────────
obs = env.reset()
print("\n" + "=" * 60)
print("RESET OUTPUT")
print("=" * 60)
print(f"type      : {type(obs)}")
print(f"obs.shape : {obs.shape}")
print(f"obs       : {obs}")

# ── 7. Do a step and inspect raw result + info ───────────────────────
action = env.action_space.sample()
result = env.step(action)

print("\n" + "=" * 60)
print("STEP OUTPUT")
print("=" * 60)
print(f"result length : {len(result)}")
print(f"action        : {action}")
print(f"obs.shape     : {result[0].shape}")
print(f"obs           : {result[0]}")
print(f"reward        : {result[1]}")
print(f"done          : {result[2]}")
print(f"info keys     : {list(result[3].keys())}")
print(f"info          : {result[3]}")

# ── 8. Print the gym agent source to understand raw_state ────────────
print("\n" + "=" * 60)
print("FINANCIAL GYM AGENT SOURCE (raw_state keys)")
print("=" * 60)
try:
    from abides_gym.experimental_agents.financial_gym_agent import FinancialGymAgent
    print(inspect.getsource(FinancialGymAgent))
except Exception as e:
    print(f"Could not load FinancialGymAgent: {e}")

# ── 9. Print order size model to understand action encoding ──────────
print("\n" + "=" * 60)
print("ORDER SIZE MODEL SOURCE")
print("=" * 60)
try:
    from abides_markets.models.order_size_model import OrderSizeModel
    print(inspect.getsource(OrderSizeModel))
except Exception as e:
    print(f"Could not load OrderSizeModel: {e}")

# import numpy as np
# from envs.lob_env import LOBMarketMakingEnv

# def check_lob_env(env, n_steps=50):
#     print(f"=== Checking {env.__class__.__name__} | reward_type={env.reward_type} ===")

#     # ── Reset ─────────────────────────────────────────────────────────
#     obs, info = env.reset(seed=42)

#     assert isinstance(obs, np.ndarray),          f"obs must be ndarray, got {type(obs)}"
#     assert obs.dtype == np.float32,              f"obs dtype must be float32, got {obs.dtype}"
#     assert obs.shape == env.observation_space.shape, \
#         f"obs shape {obs.shape} != obs space {env.observation_space.shape}"
#     assert not np.any(np.isnan(obs)),            "obs contains NaN after reset"
#     assert not np.any(np.isinf(obs)),            "obs contains Inf after reset"
#     assert isinstance(info, dict),               f"info must be dict, got {type(info)}"
#     print(f"  reset OK | obs.shape={obs.shape} info.keys={list(info.keys())}")

#     # ── Step loop ─────────────────────────────────────────────────────
#     rewards = []
#     for i in range(n_steps):
#         action = env.action_space.sample()
#         assert env.action_space.contains(action), f"sampled action not in action_space: {action}"

#         result = env.step(action)
#         assert len(result) == 5, f"step must return 5 values, got {len(result)}"
#         obs, reward, terminated, truncated, info = result

#         # obs
#         assert isinstance(obs, np.ndarray),               "step obs must be ndarray"
#         assert obs.dtype == np.float32,                   f"step obs dtype must be float32, got {obs.dtype}"
#         assert obs.shape == env.observation_space.shape,  f"step obs shape {obs.shape} != {env.observation_space.shape}"
#         assert not np.any(np.isnan(obs)),                 f"obs contains NaN at step {i}"
#         assert not np.any(np.isinf(obs)),                 f"obs contains Inf at step {i}"

#         # reward
#         assert isinstance(reward, float),                 f"reward must be float, got {type(reward)}"
#         assert not np.isnan(reward),                      f"reward is NaN at step {i}"
#         assert not np.isinf(reward),                      f"reward is Inf at step {i}"
#         rewards.append(reward)

#         # flags
#         assert isinstance(terminated, bool),              f"terminated must be bool, got {type(terminated)}"
#         assert isinstance(truncated, bool),               f"truncated must be bool, got {type(truncated)}"

#         # info
#         assert isinstance(info, dict),                    f"info must be dict, got {type(info)}"

#         if terminated or truncated:
#             print(f"  episode ended at step {i} | terminated={terminated} truncated={truncated}")
#             obs, info = env.reset()
#             assert obs.shape == env.observation_space.shape, "obs shape wrong after mid-loop reset"

#     print(f"  {n_steps} steps OK")
#     print(f"  reward | mean={np.mean(rewards):.4f}  std={np.std(rewards):.4f}  "
#           f"min={np.min(rewards):.4f}  max={np.max(rewards):.4f}")

#     # ── Obs space bounds (soft check — warn, don't hard fail) ─────────
#     obs, _ = env.reset()
#     violations = 0
#     for i in range(20):
#         obs, _, _, _, _ = env.step(env.action_space.sample())
#         lo = env.observation_space.low
#         hi = env.observation_space.high
#         if np.any(obs < lo) or np.any(obs > hi):
#             violations += 1
#     if violations:
#         print(f"  WARN: obs out of declared bounds in {violations}/20 steps "
#               f"(expected with -inf/+inf box)")
#     else:
#         print(f"  obs bounds OK")

#     # ── Second reset reproduces valid obs ─────────────────────────────
#     obs2, _ = env.reset(seed=42)
#     assert obs2.shape == env.observation_space.shape, "second reset shape wrong"
#     assert not np.any(np.isnan(obs2)), "NaN after second reset"
#     print(f"  second reset OK")
#     print(f"=== PASS ===\n")


# if __name__ == "__main__":
#     for reward_type in ("asymmetric", "quadratic", "sparse"):
#         env = LOBMarketMakingEnv(reward_type=reward_type, episode_len=100, seed=0)
#         check_lob_env(env, n_steps=50)
#         env.close()

#-------------------------------------------------------------------------------
# from envs.lob_env import LOBMarketMakingEnv
# from envs.stylized_facts import (
#     check_volatility_clustering,
#     check_spread_autocorrelation, 
#     check_queue_imbalance_predictability,
# )
# import numpy as np

# env = LOBMarketMakingEnv(reward_type="asymmetric", episode_len=390, seed=42)

# all_returns    = []
# all_spreads    = []
# all_imbalances = []
# all_future_moves = []

# for ep in range(5):
#     obs, info = env.reset(seed=ep)
#     mid_prices = [info["mid_price"]]
#     spreads    = []
#     imbalances = []

#     done = False
#     while not done:
#         obs, r, term, trunc, info = env.step(env.action_space.sample())
#         mid_prices.append(info["mid_price"])
#         spreads.append(info.get("market_spread", info["ask_price"] - info["bid_price"]))
#         imbalances.append(info["queue_imbalance"])
#         done = term or trunc

#     returns = np.diff(np.log(np.maximum(np.array(mid_prices), 1e-10)))
#     all_returns.extend(returns.tolist())
#     all_spreads.extend(spreads)

#     if len(imbalances) == len(returns):
#         all_imbalances.extend(imbalances)
#         all_future_moves.extend(returns.tolist())

# env.close()

# print(f"Total returns:    {len(all_returns)}")
# print(f"Total spreads:    {len(all_spreads)}")
# print(f"Total imbalances: {len(all_imbalances)}")

# p2, s2 = check_volatility_clustering(np.array(all_returns))
# print(f"\nVol clustering:   passed={p2}  mean_autocorr={s2['mean_autocorr']:.4f}")

# p3, s3 = check_spread_autocorrelation(np.array(all_spreads))
# print(f"Spread autocorr:  passed={p3}  lag1={s3['lag1_autocorr']:.4f}")

# p5, s5 = check_queue_imbalance_predictability(
#     np.array(all_imbalances), np.array(all_future_moves)
# )
# print(f"Queue imbalance:  passed={p5}  corr={s5['correlation']:.4f}")

# # Raw signal check
# print(f"\nSpread unique values: {len(np.unique(np.round(all_spreads, 4)))}")
# print(f"Spread std: {np.std(all_spreads):.4f}")
# print(f"|returns| mean autocorr at lag 1: {np.corrcoef(np.abs(all_returns[:-1]), np.abs(all_returns[1:]))[0,1]:.4f}")