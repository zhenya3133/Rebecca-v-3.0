from platform_logger.platform_logger_facade import log_event


def run_agent(context, input_data):
    try:
        log_event(f"{__name__}: started with data {input_data}")
        memory = context["memory"]
        semantic = memory.semantic
        episodic = memory.episodic
        procedural = memory.procedural
        vault = memory.vault
        semantic.store_concept("knowledge_asset", input_data)
        episodic.store_event({"type": "knowledge_ingest", "payload": input_data})
        procedural.store_workflow("knowledge_curator", ["ingest", "normalize", "link"])
        vault.store_secret("knowledge_curator", str(input_data))
        log_event(f"{__name__}: completed successfully")
        return {"result": "knowledge curated", "context": context}
    except Exception as exc:
        log_event(f"{__name__}: error - {exc}")
        return {"result": None, "error": str(exc), "context": context}
