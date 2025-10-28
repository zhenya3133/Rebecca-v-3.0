"""Smoke coverage for orchestrator main workflow."""

from orchestrator.main_workflow import main_workflow


def test_main_workflow_enriches_memory_layers():
    task = {"task": "smoke"}
    result = main_workflow(task)

    assert result["result"] == "ui_ux updated"
    context = result["context"]
    memory = context["memory"]

    assert memory.core.get_fact("architecture") == "initialized"
    blueprint = memory.core.get_fact("blueprint")
    assert blueprint and blueprint["modules"] == ["ingest", "memory", "analytics"]
    assert memory.semantic.get_concept("research_topic") == "AI research"
    assert memory.semantic.get_concept("ui element") == "button"
    assert memory.procedural.get_workflow("deploy") == ["build", "test", "deploy"]
    assert memory.episodic.get_events(), "Episodic memory should capture events"
    assert memory.security.get_audits(), "Security audits must be recorded"
    assert memory.vault.get_secret("api_key") == "example_api_key"
