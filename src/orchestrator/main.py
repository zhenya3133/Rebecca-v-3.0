from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        episodic = context["memory"].episodic
        core = context["memory"].core
        episodic.store_event("workflow run")
        core.store_fact("orchestrator", True)
        log_event(f"{__name__}: completed successfully")
        return {"result": "orchestrator complete", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
