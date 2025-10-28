from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        semantic = context["memory"].semantic
        episodic = context["memory"].episodic
        semantic.store_concept("ui element", "button")
        episodic.store_event("ui interaction")
        log_event(f"{__name__}: completed successfully")
        return {"result": "ui_ux updated", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
