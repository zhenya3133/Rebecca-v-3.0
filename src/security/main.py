from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        security_mem = context["memory"].security
        vault = context["memory"].vault
        security_mem.store_audit("checked user access")
        vault.store_secret("root_password", "changeme")
        log_event(f"{__name__}: completed successfully")
        return {"result": "security audit", "context": context}
    except Exception as e:
        log_event(f"{__name__}: error - {str(e)}")
        return {"result": None, "error": str(e), "context": context}
