"""Setting keys for summary-mode fetch prompts.

The actual template text lives in ``defaults/settings_prompts.json`` and is
resolved at call time so per-user settings and ``LDR_PROMPTS_...`` overrides
apply immediately.
"""

SUMMARY_FOCUS_PROMPT_KEY = (
    "prompts.advanced_search_system.tools.fetch.summary_focus"
)
SUMMARY_FOCUS_QUERY_PROMPT_KEY = (
    "prompts.advanced_search_system.tools.fetch.summary_focus_query"
)

__all__ = [
    "SUMMARY_FOCUS_PROMPT_KEY",
    "SUMMARY_FOCUS_QUERY_PROMPT_KEY",
]
