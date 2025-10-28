from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        procedural = context["memory"].procedural
        episodic = context["memory"].episodic
        procedural.store_workflow("schedule", ["plan", "allocate", "execute"])
        episodic.store_event("scheduler: new round")
        log_event(f"{__name__}: completed successfully")
        return {"result": "scheduler run", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
