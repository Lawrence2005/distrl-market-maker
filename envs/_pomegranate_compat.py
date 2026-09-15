"""
envs/_pomegranate_compat.py

Minimal stand-in for the one piece of the legacy pomegranate<1.0 API that
the vendored ABIDES fork depends on:
abides_markets.models.order_size_model.OrderSizeModel does

    from pomegranate import GeneralMixtureModel
    self.model = GeneralMixtureModel.from_json(json.dumps(order_size))
    self.model.sample(random_state=...)

The `pomegranate` on PyPI is now a from-scratch PyTorch rewrite (by a
different maintainer, same package name) that has no `GeneralMixtureModel`
at all, let alone `.from_json`/`.sample`; the old 0.14.x release the fork
was written against can't be installed on Python 3.12 (its Cython sources
predate the 3.12 build toolchain — confirmed by attempting the build).

Rather than patch the vendored abides-jpmc-public package (see CLAUDE.md —
project-specific fixes belong in envs/, not the vendored fork), this
reimplements just that one call site: a JSON mixture of Normal/LogNormal
components with weighted sampling. envs/lob_env.py installs this into
sys.modules["pomegranate"] before anything imports abides_markets.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np


class _Component:
    def __init__(self, name: str, parameters: Sequence[float]):
        self.name = name
        self.parameters = parameters

    def sample(self, random_state: np.random.RandomState) -> float:
        mu, sigma = self.parameters
        if self.name == "NormalDistribution":
            return random_state.normal(mu, sigma)
        if self.name == "LogNormalDistribution":
            return random_state.lognormal(mu, sigma)
        raise ValueError(
            f"pomegranate compat shim: unsupported distribution '{self.name}'"
        )


class GeneralMixtureModel:
    """Reimplements pomegranate<1.0's JSON deserialization + weighted-
    component sampling — the only two operations order_size_model.py uses."""

    def __init__(self, components: list[_Component], weights: Sequence[float]):
        self._components = components
        w = np.asarray(weights, dtype=np.float64)
        self._weights = w / w.sum()

    @classmethod
    def from_json(cls, s: str) -> "GeneralMixtureModel":
        data = json.loads(s)
        components = [
            _Component(d["name"], d["parameters"]) for d in data["distributions"]
        ]
        return cls(components, data["weights"])

    def sample(self, random_state: np.random.RandomState) -> float:
        idx = random_state.choice(len(self._components), p=self._weights)
        return self._components[idx].sample(random_state)
