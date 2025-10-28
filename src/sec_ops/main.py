from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        security = context["memory"].security
        security.register_alert("sec_ops_scan", {"target": input_data})
        log_event(f"{__name__}: completed successfully")
        return {"result": "security review complete", "context": context}
    except Exception as exc:
        log_event(f"{__name__}: error - {exc}")
        return {"result": None, "error": str(exc), "context": context}
