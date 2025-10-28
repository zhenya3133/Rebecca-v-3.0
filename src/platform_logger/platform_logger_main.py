from .platform_logger_writer import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        episodic = context["memory"].episodic
        security_mem = context["memory"].security
        episodic.store_event("logger event")
        security_mem.store_audit("logging audit")
        log_event(f"{__name__}: completed successfully")
        return {"result": "logger complete", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
