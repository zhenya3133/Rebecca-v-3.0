from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        episodic = context["memory"].episodic
        procedural = context["memory"].procedural
        episodic.store_event("qa check triggered")
        procedural.store_workflow("test_case", ["open app", "validate output"])
        log_event(f"{__name__}: completed successfully")
        return {"result": "qa review", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
