from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        procedural = context["memory"].procedural
        deployment_plan = {
            "environments": ["dev", "staging", "prod"],
            "strategy": "blue-green",
            "inputs": input_data,
        }
        procedural.store_workflow("deployment_ops", deployment_plan["environments"])
        log_event(f"{__name__}: completed successfully")
        return {"result": deployment_plan, "context": context}
    except Exception as exc:
        log_event(f"{__name__}: error - {exc}")
        return {"result": None, "error": str(exc), "context": context}
