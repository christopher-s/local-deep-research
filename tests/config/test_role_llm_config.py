"""Tests for role-specific LLM configuration."""

from unittest.mock import MagicMock, patch

from local_deep_research.config.llm_config import (
    get_role_llm,
    resolve_base_llm_settings,
    resolve_role_llm_settings,
)


def _base_snapshot():
    return {
        "search.tool": "searxng",
        "llm.provider": "openai_endpoint",
        "llm.model": "qwen-search",
        "llm.temperature": 0.2,
        "llm.max_tokens": 4000,
    }


def test_analysis_role_overrides_base_llm_settings():
    snapshot = {
        **_base_snapshot(),
        "llm.analysis.provider": "openrouter",
        "llm.analysis.model": "analysis-model",
        "llm.analysis.temperature": 0.1,
        "llm.analysis.max_tokens": 8000,
    }

    resolved = resolve_role_llm_settings("analysis", snapshot)

    assert resolved == {
        "provider": "openrouter",
        "model": "analysis-model",
        "temperature": 0.1,
        "max_tokens": 8000,
    }


def test_base_settings_use_effective_request_overrides():
    resolved = resolve_base_llm_settings(
        _base_snapshot(),
        provider="anthropic",
        model_name="claude-search",
        temperature=0.35,
    )

    assert resolved == {
        "provider": "anthropic",
        "model": "claude-search",
        "temperature": 0.35,
        "max_tokens": 4000,
    }


def test_report_partial_override_inherits_effective_search_override():
    snapshot = {**_base_snapshot(), "llm.report.model": "report-model"}
    base = resolve_base_llm_settings(
        snapshot, provider="anthropic", model_name="claude-search"
    )
    analysis = resolve_role_llm_settings(
        "analysis", snapshot, fallback_settings=base
    )

    report = resolve_role_llm_settings(
        "report", snapshot, fallback_settings=analysis
    )

    assert report["provider"] == "anthropic"
    assert report["model"] == "report-model"


def test_report_role_inherits_analysis_settings_when_not_overridden():
    snapshot = {
        **_base_snapshot(),
        "llm.analysis.provider": "openrouter",
        "llm.analysis.model": "analysis-model",
        "llm.report.model": "report-model",
    }
    analysis = resolve_role_llm_settings("analysis", snapshot)

    report = resolve_role_llm_settings(
        "report", snapshot, fallback_settings=analysis
    )

    assert report == {
        "provider": "openrouter",
        "model": "report-model",
        "temperature": 0.2,
        "max_tokens": 4000,
    }


def test_role_without_overrides_reuses_fallback_llm():
    fallback_llm = MagicMock(name="search_llm")

    with patch("local_deep_research.config.llm_config.get_llm") as get_llm_mock:
        llm, settings, owns_llm = get_role_llm(
            "analysis",
            fallback_llm=fallback_llm,
            settings_snapshot=_base_snapshot(),
        )

    assert llm is fallback_llm
    assert settings["model"] == "qwen-search"
    assert owns_llm is False
    get_llm_mock.assert_not_called()


def test_role_override_builds_llm_with_derived_snapshot():
    snapshot = {
        **_base_snapshot(),
        "llm.analysis.provider": "openrouter",
        "llm.analysis.model": "analysis-model",
        "llm.analysis.max_tokens": 12000,
    }
    fallback_llm = MagicMock(name="search_llm")
    created_llm = MagicMock(name="analysis_llm")

    with patch(
        "local_deep_research.config.llm_config.get_llm",
        return_value=created_llm,
    ) as get_llm_mock:
        llm, settings, owns_llm = get_role_llm(
            "analysis",
            fallback_llm=fallback_llm,
            settings_snapshot=snapshot,
            research_id="research-1",
            research_context={"phase": "analysis"},
        )

    assert llm is created_llm
    assert owns_llm is True
    assert settings["model"] == "analysis-model"

    call = get_llm_mock.call_args
    assert call.kwargs["provider"] == "openrouter"
    assert call.kwargs["model_name"] == "analysis-model"
    derived_snapshot = call.kwargs["settings_snapshot"]
    assert derived_snapshot["llm.provider"] == "openrouter"
    assert derived_snapshot["llm.model"] == "analysis-model"
    assert derived_snapshot["llm.max_tokens"] == 12000
    assert derived_snapshot["search.tool"] == "searxng"
