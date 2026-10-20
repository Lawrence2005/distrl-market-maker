# tests/ — Unit Tests

Run all tests: `pytest tests/`

## Coverage targets
- baselines/: AS and GLFT quote calculations verified against analytical solutions
- agents/cvar_policy.py: CVaR wrapper tested on known toy distribution
- evaluation/metrics.py: Sharpe, MAP, CVaR verified against numpy reference
- envs/: Environment step() returns correct shape, reward in expected range
- envs/fine_grained.py + MarketDataConfig: recording leaves the simulation unchanged, per-second frame rows at decision times equal the agent's observation (`test_fine_grained.py`; ABIDES tests marked `abides`)
