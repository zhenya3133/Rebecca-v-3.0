import os
import sys
from datetime import datetime, UTC

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from observability.audit_log import AuditLogger


def test_audit_logger_append(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_file)
    payload = {"ts": datetime.now(UTC).isoformat(), "event": "test"}
    logger.write(payload)
    with log_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert lines and "test" in lines[0]
