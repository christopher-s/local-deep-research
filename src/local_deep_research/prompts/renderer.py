"""Safe rendering for editable runtime prompts.

Prompt templates are ordinary settings under the ``prompts.*`` namespace.
The shipped value in ``defaults/settings_prompts.json`` is the fallback, while
user/database values and ``LDR_PROMPTS_...`` environment overrides are resolved
through the standard settings path.

Templates use explicit ``{{placeholder}}`` tokens. Substitution is deliberately
non-executable: prompt text cannot evaluate Python expressions or access object
attributes.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

from loguru import logger

from ..config.thread_settings import get_setting_from_snapshot
from ..settings.manager import check_env_setting

_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_PROMPT_SETTINGS_FILE = "settings_prompts.json"


class PromptRenderError(ValueError):
    """Raised when an editable prompt cannot be rendered safely."""


@lru_cache(maxsize=1)
def _prompt_catalog() -> dict[str, dict[str, Any]]:
    path = resources.files("local_deep_research.defaults").joinpath(
        _PROMPT_SETTINGS_FILE
    )
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"{_PROMPT_SETTINGS_FILE} must contain a JSON object")
    return data


def get_prompt_default(key: str) -> str:
    """Return the shipped default template for *key*."""
    item = _prompt_catalog().get(key)
    if not item:
        raise KeyError(f"Unknown prompt setting: {key}")
    value = item.get("value")
    if not isinstance(value, str):
        raise TypeError(f"Prompt setting {key} must have a string default")
    return value


def get_prompt_template(
    key: str,
    *,
    settings_snapshot: Mapping[str, Any] | None = None,
) -> str:
    """Resolve a prompt through the normal settings/environment precedence."""
    default = get_prompt_default(key)

    # Environment variables are the operator-level override and must work even
    # in programmatic paths that have no per-user settings context. Prompt
    # values are strings, so no additional type conversion is required.
    env_value = check_env_setting(key)
    if env_value is not None:
        return env_value

    value = get_setting_from_snapshot(
        key,
        default=default,
        settings_snapshot=settings_snapshot,
    )
    if value is None:
        return default
    if not isinstance(value, str):
        logger.warning(
            "Prompt setting '{}' resolved to {}; using shipped default",
            key,
            type(value).__name__,
        )
        return default
    return value


def _render_template(key: str, template: str, values: Mapping[str, Any]) -> str:
    referenced = set(_PLACEHOLDER_RE.findall(template))
    missing = sorted(name for name in referenced if name not in values)
    if missing:
        raise PromptRenderError(
            f"Prompt {key} is missing values for: {', '.join(missing)}"
        )

    def replace(match: re.Match[str]) -> str:
        value = values[match.group(1)]
        return "" if value is None else str(value)

    return _PLACEHOLDER_RE.sub(replace, template)


def render_prompt(
    key: str,
    *,
    settings_snapshot: Mapping[str, Any] | None = None,
    **values: Any,
) -> str:
    """Render an editable prompt safely.

    A malformed user override falls back to the shipped template so a typo in
    Settings cannot break research execution. Unknown placeholders in the
    override are treated as malformed; omitted placeholders are allowed.
    """
    default = get_prompt_default(key)
    template = get_prompt_template(key, settings_snapshot=settings_snapshot)

    try:
        return _render_template(key, template, values)
    except PromptRenderError:
        if template == default:
            raise
        logger.exception(
            "Invalid override for prompt '{}'; falling back to shipped template",
            key,
        )
        return _render_template(key, default, values)
