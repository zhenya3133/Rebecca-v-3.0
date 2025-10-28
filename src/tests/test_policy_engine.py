import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from policy_engine.policy_engine import PolicyEngine
from policy_engine.evaluators import allow_all
from schema.nodes import Fact


def test_policy_allow_all():
    fact = Fact(
        id="1",
        ntype="Fact",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        owner="user",
        privacy="team",
        confidence=0.8,
        subject="A",
        predicate="is",
        object="B",
        evidence=[],
    )
    engine = PolicyEngine([allow_all])
    assert engine.allowed("actor", fact, "read")
    print("Policy allow_all OK")
