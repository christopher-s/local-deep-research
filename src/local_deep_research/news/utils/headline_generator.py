"""
Headline generation utilities for news items.
Uses LLM to generate concise, meaningful headlines from long queries and findings.
"""

from typing import Optional
from loguru import logger
from local_deep_research.prompts import render_prompt


def generate_headline(
    query: str,
    findings: str = "",
    max_length: int = 100,
    settings_snapshot: Optional[dict] = None,
) -> str:
    """
    Generate a concise headline from a query and optional findings.

    Args:
        query: The search query or research question
        findings: Optional findings/content to help generate better headline
        max_length: Maximum length for the headline
        settings_snapshot: Optional settings snapshot so the LLM call
            picks up the active egress policy. Background callers
            should pass this through; without it, the LLM PEP's
            ``settings_snapshot is not None`` guard skips and a cloud
            LLM can fire even under require_local_endpoint.

    Returns:
        A concise headline string
    """
    # Always try LLM generation first for dynamic headlines based on actual content
    llm_headline = _generate_with_llm(
        query, findings, max_length, settings_snapshot
    )
    if llm_headline:
        return llm_headline

    # No fallback - if LLM fails, indicate failure
    return "[Headline generation failed]"


def _generate_with_llm(
    query: str,
    findings: str,
    max_length: int,
    settings_snapshot: Optional[dict] = None,
) -> Optional[str]:
    """Generate headline using LLM."""
    try:
        from ...config.llm_config import get_llm

        # Use the configured model for headline generation
        llm = get_llm(temperature=0.3, settings_snapshot=settings_snapshot)

        try:
            # Focus only on the findings/report content, not the query
            if not findings:
                logger.debug("No findings provided for headline generation")
                return None

            # Use the COMPLETE findings - no character limit
            findings_preview = findings
            logger.debug(
                f"Generating headline with {len(findings)} chars of findings"
            )

            prompt = render_prompt(
                "prompts.news.utils.headline_generator.generate_with_llm.prompt",
                findings_preview=findings_preview,
            )

            response = llm.invoke(prompt)
            headline: str = str(response.content).strip()

            # Clean up the generated headline
            headline = headline.strip("\"'.,!?")

            # Validate the headline
            if headline:
                logger.debug(f"Generated headline: {headline}")
                return headline
        finally:
            from ...utilities.resource_utils import safe_close

            safe_close(llm, "headline LLM")

    except Exception as e:
        logger.debug(f"LLM headline generation failed: {e}")

    return None
