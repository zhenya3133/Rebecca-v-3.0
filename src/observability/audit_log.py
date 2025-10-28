"""Audit logging utilities with in-memory friendly design."""

import datetime
from datetime import UTC
import hashlib
import json
from pathlib import Path
from typing import Iterable


class AuditLogger:
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)

    def write(self, record: dict) -> str:
        payload = json.dumps(record, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        enriched = {**record, "hash": digest}
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, sort_keys=True) + "\n")
        return digest


def audit(event_type: str, payload: dict, actor: str, blockchain_log: bool = False):
    ts = datetime.datetime.now(UTC).isoformat()
    data = {
        "ts": ts,
        "actor": actor,
        "event_type": event_type,
        "payload": payload,
    }
    logger = AuditLogger(Path(__file__).parent / "audit_log.jsonl")
    digest = logger.write(data)
    if blockchain_log:
        chain_file = Path(__file__).parent / "onchain_audit_hashes.txt"
        chain_file.parent.mkdir(parents=True, exist_ok=True)
        with chain_file.open("a", encoding="utf-8") as f:
            f.write(digest + "\n")
    return digest
