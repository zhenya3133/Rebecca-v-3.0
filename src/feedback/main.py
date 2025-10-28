from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        episodic = context["memory"].episodic
        semantic = context["memory"].semantic
        episodic.store_event("user feedback")
        semantic.store_concept("feedback", input_data)
        log_event(f"{__name__}: completed successfully")
        return {"result": "feedback saved", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
