from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        qa_plan = {
            "tests": ["unit", "integration", "security"],
            "focus": input_data,
        }
        context["memory"].episodic.store_event({"type": "qa_plan", "payload": qa_plan})
        log_event(f"{__name__}: completed successfully")
        return {"result": qa_plan, "context": context}
    except Exception as exc:
        log_event(f"{__name__}: error - {exc}")
        return {"result": None, "error": str(exc), "context": context}
