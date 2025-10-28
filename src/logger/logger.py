from datetime import datetime


def log_event(msg):
    with open("agent_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
