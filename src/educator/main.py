from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        core = context["memory"].core
        semantic = context["memory"].semantic
        core.store_fact("lesson", input_data)
        semantic.store_concept("education topic", "machine learning")
        log_event(f"{__name__}: completed successfully")
        return {"result": "educator complete", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
