from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        procedural = context["memory"].procedural
        core = context["memory"].core
        tracker = context["memory"].blueprint_tracker
        blueprint = {
            "modules": ["ingest", "memory", "analytics"],
            "pipelines": ["collect", "enrich", "deploy"],
        }
        procedural.store_workflow("blueprint_generator", blueprint["pipelines"])
        core.store_fact("blueprint", blueprint)
        tracker.record_blueprint(blueprint)
        log_event(f"{__name__}: completed successfully")
        return {"result": blueprint, "context": context}
    except Exception as exc:
        log_event(f"{__name__}: error - {exc}")
        return {"result": None, "error": str(exc), "context": context}
