from schema.nodes import NodeBase


def redact(node: NodeBase) -> NodeBase:
    if node.attrs.get("pii") == "yes":
        safe_attrs = {
            k: v for k, v in node.attrs.items() if k not in {"email", "phone"}
        }
        return node.copy(update={"attrs": {**safe_attrs, "redacted": "true"}})
    return node
