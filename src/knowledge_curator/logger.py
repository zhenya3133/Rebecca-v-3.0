from platform_logger.platform_logger_facade import log_event


def log(message: str) -> None:
    log_event(f"knowledge_curator: {message}")
