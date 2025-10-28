from audit_log import audit


def test_audit_log():
    h = audit("test_event", {"foo": "bar"}, actor="system", blockchain_log=True)
    assert len(h) == 64
    print("audit_log OK:", h)


if __name__ == "__main__":
    test_audit_log()
