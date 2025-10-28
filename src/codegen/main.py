from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        procedural = context["memory"].procedural
        vault = context["memory"].vault
        procedural.store_workflow("deploy", ["build", "test", "deploy"])
        vault.store_secret("api_key", "example_api_key")
        log_event(f"{__name__}: completed successfully")
        return {"result": "codegen complete", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
