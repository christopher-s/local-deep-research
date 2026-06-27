"""Coverage guards for the editable prompt catalogue and settings UI."""

import ast
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src/local_deep_research"
CATALOG_PATH = SOURCE_ROOT / "defaults/settings_prompts.json"
PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def _catalog():
    return json.loads(CATALOG_PATH.read_text())


def test_catalog_entries_are_editable_multiline_settings():
    catalog = _catalog()
    assert len(catalog) >= 80
    for key, item in catalog.items():
        assert key.startswith("prompts.")
        assert item["editable"] is True
        assert item["visible"] is True
        assert item["ui_element"] == "textarea"
        assert item["type"] == "APP"
        assert isinstance(item["value"], str) and item["value"].strip()
        assert item["category"].startswith("prompts_")


def test_every_prompt_default_can_render_with_declared_placeholders(monkeypatch):
    from local_deep_research.prompts.renderer import render_prompt

    catalog = _catalog()
    for key, item in catalog.items():
        env_name = "LDR_" + "_".join(key.split(".")).upper()
        monkeypatch.delenv(env_name, raising=False)
        placeholders = set(PLACEHOLDER_RE.findall(item["value"]))
        values = {name: f"<{name}>" for name in placeholders}
        rendered = render_prompt(key, **values)  # type: ignore[arg-type]
        assert isinstance(rendered, str)
        assert not PLACEHOLDER_RE.search(rendered), key


def test_all_literal_render_prompt_keys_exist_in_catalog():
    catalog = _catalog()
    referenced = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "render_prompt":
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                referenced.add(node.args[0].value)
    assert referenced <= set(catalog)


def test_every_catalog_key_is_referenced_by_runtime_source():
    catalog = _catalog()
    source_text = "\n".join(
        path.read_text(errors="ignore") for path in SOURCE_ROOT.rglob("*.py")
    )
    unreferenced = [key for key in catalog if key not in source_text]
    assert not unreferenced


def test_no_direct_literal_prompt_is_passed_to_llm_calls():
    violations = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name not in {"invoke", "ainvoke", "create_agent"}:
                continue
            for argument in node.args:
                if isinstance(argument, ast.JoinedStr) or (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                ):
                    violations.append((str(path.relative_to(PROJECT_ROOT)), node.lineno))
    assert not violations


def test_prompts_tab_is_present_and_renderable():
    dashboard = (SOURCE_ROOT / "web/templates/settings_dashboard.html").read_text()
    javascript = (SOURCE_ROOT / "web/static/js/components/settings.js").read_text()
    routes = (SOURCE_ROOT / "web/routes/settings_routes.py").read_text()

    assert 'data-tab="prompts"' in dashboard
    assert "'prompts': 'Prompts'" in javascript
    assert "'prompts'" in javascript
    assert '"prompts."' in routes
