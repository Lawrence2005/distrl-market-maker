# tests/ — Unit Tests

Run all tests: `pytest tests/`

## Coverage targets
- baselines/: AS and GLFT quote calculations verified against analytical solutions
- agents/cvar_policy.py: CVaR wrapper tested on known toy distribution
- evaluation/metrics.py: Sharpe, MAP, CVaR verified against numpy reference
- envs/: Environment step() returns correct shape, reward in expected range
