from typing import Iterable, List

from policy_engine.policy_engine import PolicyEngine
from policy_engine.transformers import redact
from schema.context_pack import ContextPack
from schema.nodes import NodeBase


def build_context_pack(
    query: str,
    nodes: Iterable[NodeBase],
    policy: PolicyEngine,
    actor: str,
    budget_tokens: int,
) -> ContextPack:
    selected: List[NodeBase] = []
    rationale: List[str] = []
    denied: List[str] = []

    for node in nodes:
        if policy.allowed(actor, node, "read"):
            sanitized = redact(node)
            selected.append(sanitized)
            rationale.append(f"match:{sanitized.ntype} conf={sanitized.confidence:.2f}")
        else:
            denied.append(node.id)

    return ContextPack(
        query=query,
        nodes=selected,
        policies_applied=denied,
        rationale=rationale,
        token_budget=budget_tokens,
    )
