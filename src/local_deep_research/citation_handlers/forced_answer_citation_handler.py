"""
Forced answer citation handler - optimized for BrowseComp-style questions.
Always provides a specific answer, never returns "cannot determine".
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from loguru import logger

from .base_citation_handler import BaseCitationHandler
from local_deep_research.prompts import render_prompt


class ForcedAnswerCitationHandler(BaseCitationHandler):
    """Citation handler that forces direct answers for benchmark questions."""

    def analyze_initial(
        self, query: str, search_results: Union[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Initial analysis with forced answer generation."""
        documents = self._create_documents(search_results)
        formatted_sources = self._format_sources(documents)

        current_timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )

        output_prefix = self._get_output_instruction_prefix()

        prompt = render_prompt(
            "prompts.citation_handlers.forced_answer_citation_handler.forcedanswercitationhandler.analyze_initial.prompt",
            output_prefix=output_prefix,
            query=query,
            formatted_sources=formatted_sources,
            current_timestamp=current_timestamp,
        )

        response = self._invoke_with_streaming(prompt)

        # If the response still doesn't have a direct answer, extract one
        if self._needs_answer_extraction(response, query):
            response = self._extract_direct_answer(
                query, response, formatted_sources
            )

        return {"content": response, "documents": documents}

    def analyze_followup(
        self,
        question: str,
        search_results: Union[str, List[Dict]],
        previous_knowledge: str,
        nr_of_links: int,
    ) -> Dict[str, Any]:
        """Follow-up analysis with forced answer generation."""
        documents = self._create_documents(
            search_results, nr_of_links=nr_of_links
        )
        formatted_sources = self._format_sources(documents)

        # Fact-checking step (if enabled)
        fact_check_response = ""
        if self.is_fact_checking_enabled():
            fact_check_prompt = render_prompt(
                "prompts.citation_handlers.forced_answer_citation_handler.forcedanswercitationhandler.analyze_followup.fact_check_prompt",
                previous_knowledge=previous_knowledge,
                formatted_sources=formatted_sources,
            )
            fact_check_response = self._invoke_text(fact_check_prompt)

        current_timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )

        output_prefix = self._get_output_instruction_prefix()

        prompt = render_prompt(
            "prompts.citation_handlers.forced_answer_citation_handler.forcedanswercitationhandler.analyze_followup.prompt",
            output_prefix=output_prefix,
            previous_knowledge=previous_knowledge,
            question=question,
            formatted_sources=formatted_sources,
            current_timestamp=current_timestamp,
            fact_check_response=fact_check_response,
        )

        content = self._invoke_with_streaming(prompt)

        # Final check - if still no direct answer, force extraction
        if self._needs_answer_extraction(content, question):
            content = self._extract_direct_answer(
                question, content, formatted_sources
            )
            logger.info(f"Forced answer extraction applied: {content[:100]}...")

        return {"content": content, "documents": documents}

    def _needs_answer_extraction(self, content: str, query: str) -> bool:
        """Check if the response needs forced answer extraction."""
        no_answer_indicators = [
            "cannot determine",
            "unable to find",
            "insufficient",
            "unclear",
            "not enough",
            "cannot provide",
            "no specific answer",
            "cannot definitively",
        ]

        content_lower = content.lower()

        # Check for no-answer indicators
        for indicator in no_answer_indicators:
            if indicator in content_lower:
                return True

        # Check if it's a direct question but no direct answer given
        if query.lower().startswith(
            ("what", "who", "which", "where", "when", "name")
        ):
            # Look for a direct answer pattern in first 100 chars
            first_part = content[:100].lower()
            if not any(
                word in first_part for word in ["is", "was", "are", "were", ":"]
            ):
                return True

        return False

    def _extract_direct_answer(
        self, query: str, content: str, sources: str
    ) -> str:
        """Force extraction of a direct answer using LLM."""
        extraction_prompt = render_prompt(
            "prompts.citation_handlers.forced_answer_citation_handler.forcedanswercitationhandler.extract_direct_answer.extraction_prompt",
            query=query,
            content_excerpt=content[:1500],
            sources_excerpt=sources[:1500],
        )

        try:
            answer = self._invoke_text(extraction_prompt)
            if not answer:
                return content

            # Format as a proper response
            return f"{answer}. Based on the available sources, this appears to be the most likely answer. {content}"

        except Exception:
            logger.exception("Error in forced answer extraction")
            # Fallback - just prepend a guess
            return f"Based on the available evidence, the most likely answer appears to be related to the search results. {content}"
