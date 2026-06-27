"""Prompt-setting persistence metadata tests."""

from types import SimpleNamespace

from local_deep_research.database.models.settings import SettingType
from local_deep_research.web.routes.settings_routes import (
    _build_new_setting_payload,
)


def test_first_prompt_edit_preserves_default_metadata():
    key = "prompts.benchmarks.simpleqa_grader"
    manager = SimpleNamespace(
        default_settings={
            key: {
                "type": "APP",
                "name": "Benchmarks / SimpleQA grader",
                "description": "Editable evaluator prompt",
                "category": "prompts_benchmarks",
                "ui_element": "textarea",
                "options": None,
                "min_value": None,
                "max_value": 100000,
                "step": None,
                "visible": True,
                "editable": True,
            }
        }
    )

    payload = _build_new_setting_payload(
        key,
        "Custom {{question}}",
        SettingType.APP,
        "prompts_general",
        manager,
    )

    assert payload["key"] == key
    assert payload["value"] == "Custom {{question}}"
    assert payload["type"] == "app"
    assert payload["ui_element"] == "textarea"
    assert payload["category"] == "prompts_benchmarks"
    assert payload["name"] == "Benchmarks / SimpleQA grader"
    assert payload["max_value"] == 100000
    assert payload["editable"] is True


def test_unknown_setting_keeps_generic_fallback_metadata():
    manager = SimpleNamespace(default_settings={})
    payload = _build_new_setting_payload(
        "app.custom_value",
        "value",
        SettingType.APP,
        "app_interface",
        manager,
    )
    assert payload["type"] == "app"
    assert payload["ui_element"] == "text"
    assert payload["category"] == "app_interface"
