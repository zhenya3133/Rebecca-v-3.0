import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from observability.metrics import (
    coverage_at_k,
    contradiction_rate,
    token_efficiency,
    drift_score,
    privacy_violation_rate,
)


class DummyPolicy:
    def allowed(self, actor, node, action):
        return node.get("allow", True)


def test_metrics_basic():
    cov = coverage_at_k(["a", "b"], ["a"], k=2)
    assert cov == 1.0
    contr = contradiction_rate(["ok", "contradiction here"])
    assert contr > 0
    eff = token_efficiency(["some text"], 10)
    assert eff > 0
    drift = drift_score([
        {"value": 0.5, "ts": drift_score.__globals__["datetime"].datetime.now()},
    ], 0.4)
    assert drift >= 0
    policy = DummyPolicy()
    rate = privacy_violation_rate([
        {"node": {"allow": True}},
        {"node": {"allow": False}},
    ], policy)
    assert rate == 0.5
