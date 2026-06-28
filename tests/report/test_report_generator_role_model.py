"""Tests for the dedicated final-report model."""

from unittest.mock import MagicMock

from local_deep_research.report_generator import IntegratedReportGenerator


def test_distinct_report_model_writes_section_from_analyzed_material():
    search_model = MagicMock(name="search_model")
    analysis_model = MagicMock(name="analysis_model")
    report_model = MagicMock(name="report_model")
    report_model.invoke.return_value = MagicMock(
        content="Final report prose preserving the finding [1]."
    )

    search_system = MagicMock(name="search_system")
    search_system.model = search_model
    search_system.analysis_model = analysis_model
    search_system.strategy.settings_snapshot = {"search.iterations": 3}
    search_system.strategy.max_iterations = 3
    search_system.analyze_topic.return_value = {
        "current_knowledge": "Analyzed evidence from the sources [1]."
    }

    generator = IntegratedReportGenerator(
        search_system=search_system,
        llm=report_model,
        settings_snapshot={"report.max_context_sections": 3},
        rewrite_sections_with_report_llm=True,
    )
    structure = [
        {
            "name": "Findings",
            "subsections": [
                {"name": "Evidence", "purpose": "Present the evidence"}
            ],
        }
    ]

    sections = generator._research_and_generate_sections(
        {"questions_by_iteration": {}}, structure, "test topic"
    )

    assert (
        "Final report prose preserving the finding [1]." in sections["Findings"]
    )
    assert "Analyzed evidence from the sources [1]." not in sections["Findings"]

    prompt = report_model.invoke.call_args.args[0]
    assert "Analyzed evidence from the sources [1]." in prompt
    assert "preserve every citation marker" in prompt.lower()
    assert "do not invent" in prompt.lower()


def test_reused_analysis_model_does_not_add_a_report_rewrite_call():
    shared_model = MagicMock(name="shared_model")
    search_system = MagicMock(name="search_system")
    search_system.model = shared_model
    search_system.analysis_model = shared_model
    search_system.strategy.settings_snapshot = {}
    search_system.strategy.max_iterations = 1
    search_system.analyze_topic.return_value = {
        "current_knowledge": "Existing synthesized prose [1]."
    }

    generator = IntegratedReportGenerator(
        search_system=search_system,
        llm=shared_model,
    )
    structure = [
        {
            "name": "Findings",
            "subsections": [
                {"name": "Evidence", "purpose": "Present the evidence"}
            ],
        }
    ]

    sections = generator._research_and_generate_sections(
        {}, structure, "test topic"
    )

    assert "Existing synthesized prose [1]." in sections["Findings"]
    shared_model.invoke.assert_not_called()
