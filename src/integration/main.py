from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        vault = context["memory"].vault
        procedural = context["memory"].procedural
        vault.store_secret("integration_token", "example_token")
        procedural.store_workflow("sync", ["connect API", "transfer data"])
        log_event(f"{__name__}: completed successfully")
        return {"result": "integration complete", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
