"""Tests for editable runtime prompt resolution and safe rendering."""

import re

import pytest

from local_deep_research.prompts.renderer import (
    PromptRenderError,
    get_prompt_default,
    get_prompt_template,
    render_prompt,
)

KEY = "prompts.benchmarks.simpleqa_grader"


def test_default_prompt_renders():
    rendered = render_prompt(
        KEY,
        question="Capital of France?",
        correct_answer="Paris",
        response="Paris",
    )
    assert "Capital of France?" in rendered
    assert "Correct Answer: Paris" in rendered
    assert "Model Response: Paris" in rendered
    assert "{{question}}" not in rendered


def test_snapshot_override_renders():
    snapshot = {
        KEY: {
            "value": "Q={{question}} | A={{correct_answer}} | R={{response}}",
            "ui_element": "textarea",
        }
    }
    assert (
        render_prompt(
            KEY,
            settings_snapshot=snapshot,
            question="Q1",
            correct_answer="A1",
            response="R1",
        )
        == "Q=Q1 | A=A1 | R=R1"
    )


def test_environment_override_has_precedence(monkeypatch):
    monkeypatch.setenv(
        "LDR_PROMPTS_BENCHMARKS_SIMPLEQA_GRADER",
        "ENV {{question}} / {{response}}",
    )
    snapshot = {KEY: {"value": "DB {{question}}", "ui_element": "textarea"}}
    assert (
        render_prompt(
            KEY,
            settings_snapshot=snapshot,
            question="question",
            correct_answer="answer",
            response="response",
        )
        == "ENV question / response"
    )


def test_empty_environment_override_is_ignored(monkeypatch):
    monkeypatch.setenv("LDR_PROMPTS_BENCHMARKS_SIMPLEQA_GRADER", "")
    snapshot = {KEY: {"value": "DB {{question}}", "ui_element": "textarea"}}
    assert (
        render_prompt(
            KEY,
            settings_snapshot=snapshot,
            question="question",
            correct_answer="answer",
            response="response",
        )
        == "DB question"
    )


def test_malformed_override_falls_back_to_shipped_default():
    snapshot = {
        KEY: {
            "value": "Broken {{unknown_placeholder}}",
            "ui_element": "textarea",
        }
    }
    rendered = render_prompt(
        KEY,
        settings_snapshot=snapshot,
        question="Capital of France?",
        correct_answer="Paris",
        response="Paris",
    )
    assert rendered != "Broken {{unknown_placeholder}}"
    assert "Capital of France?" in rendered
    assert "Correct Answer: Paris" in rendered


def test_missing_default_placeholder_raises():
    default = get_prompt_default(KEY)
    placeholders = set(re.findall(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}", default))
    assert "question" in placeholders
    with pytest.raises(PromptRenderError):
        render_prompt(KEY, correct_answer="Paris", response="Paris")


def test_environment_name_uses_standard_settings_convention(monkeypatch):
    monkeypatch.setenv("LDR_PROMPTS_BENCHMARKS_SIMPLEQA_GRADER", "operator value")
    assert get_prompt_template(KEY) == "operator value"


def test_unknown_prompt_key_raises():
    with pytest.raises(KeyError):
        render_prompt("prompts.does.not.exist")
