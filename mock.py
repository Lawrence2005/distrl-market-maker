import gymnasium as gym
import abides_gym
from pomegranate.gmm import GeneralMixtureModel

env = gym.make(
    "markets-daily_investor-v0"
)

obs, info = env.reset()

print("=== obs shape :", obs.shape)
print("=== obs values:", obs)
print("=== info keys  :", list(info.keys()))
print("=== info       :", info)

action = env.action_space.sample()
result = env.step(action)

print("\n=== step result length:", len(result))
print("=== step result:", result)