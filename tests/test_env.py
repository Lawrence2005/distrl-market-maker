"""
Unit tests for ABIDES-Gym environment wrapper.

Tests:
  - step() returns (obs, reward, done, info) with correct shapes
  - Reward is finite and within expected range
  - Inventory respects max_inventory constraint
  - Hawkes arrivals produce clustering (autocorrelation > Poisson)
"""
import pytest
# TODO: implement
