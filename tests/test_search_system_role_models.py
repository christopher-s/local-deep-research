"""Tests for separate search and analysis models in AdvancedSearchSystem."""

from unittest.mock import MagicMock

from local_deep_research.search_system import AdvancedSearchSystem


def test_search_system_uses_analysis_model_for_source_synthesis():
    search_model = MagicMock(name="search_model")
    analysis_model = MagicMock(name="analysis_model")
    search_engine = MagicMock(name="search_engine")

    system = AdvancedSearchSystem(
        llm=search_model,
        analysis_llm=analysis_model,
        search=search_engine,
        strategy_name="source-based",
        settings_snapshot={"search.tool": "searxng"},
    )

    assert system.model is search_model
    assert system.analysis_model is analysis_model
    assert system.question_generator.model is search_model
    assert system.citation_handler.llm is analysis_model
    assert system.strategy.model is search_model
    assert system.strategy.citation_handler is system.citation_handler
