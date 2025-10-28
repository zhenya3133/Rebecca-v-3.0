from typing import Callable, Iterable

from schema.nodes import NodeBase


class PolicyEngine:
    def __init__(self, rules: Iterable[Callable[[str, NodeBase, str], bool]]):
        self.rules = list(rules)

    def allowed(self, actor: str, node: NodeBase, action: str) -> bool:
        return all(rule(actor, node, action) for rule in self.rules)
