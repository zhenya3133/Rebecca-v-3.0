from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        semantic = context["memory"].semantic
        episodic = context["memory"].episodic
        semantic.store_concept("new_idea", "apply AI to analytics")
        episodic.store_event(f"generated idea: {input_data}")
        log_event(f"{__name__}: completed successfully")
        return {"result": "idea generated", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
