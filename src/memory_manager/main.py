from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        core = context["memory"].core
        procedural = context["memory"].procedural
        core.store_fact("memory access", True)
        procedural.store_workflow("memory_manager", ["init", "manage", "terminate"])
        log_event(f"{__name__}: completed successfully")
        return {"result": "memory_manager active", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
