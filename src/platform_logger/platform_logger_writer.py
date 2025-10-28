from datetime import datetime
from pathlib import Path


def log_event(msg, base_path: Path | None = None):
    target_dir = base_path or Path(__file__).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "agent_log.txt"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
