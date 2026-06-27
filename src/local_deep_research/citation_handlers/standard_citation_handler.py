"""
Standard citation handler - the original implementation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from .base_citation_handler import BaseCitationHandler
from local_deep_research.prompts import render_prompt


class StandardCitationHandler(BaseCitationHandler):
    """Standard citation handler with detailed analysis."""

    def analyze_initial(
        self, query: str, search_results: Union[str, List[Dict]]
    ) -> Dict[str, Any]:
        documents = self._create_documents(search_results)
        if not documents:
            return self._no_sources_response(query)
        formatted_sources = self._format_sources(documents)
        current_timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )

        output_prefix = self._get_output_instruction_prefix()

        prompt = render_prompt(
            "prompts.citation_handlers.standard_citation_handler.standardcitationhandler.analyze_initial.prompt",
            output_prefix=output_prefix,
            query=query,
            formatted_sources=formatted_sources,
            current_timestamp=current_timestamp,
        )

        response = self._invoke_with_streaming(prompt)
        return {"content": response, "documents": documents}

    def analyze_followup(
        self,
        question: str,
        search_results: Union[str, List[Dict]],
        previous_knowledge: str,
        nr_of_links: int,
    ) -> Dict[str, Any]:
        """Process follow-up analysis with citations."""
        documents = self._create_documents(
            search_results, nr_of_links=nr_of_links
        )
        # With previous knowledge present the LLM can still cite prior
        # sources legitimately; with neither, refuse to synthesize.
        if not documents and not (previous_knowledge or "").strip():
            return self._no_sources_response(question)
        formatted_sources = self._format_sources(documents)
        # Add fact-checking step
        fact_check_prompt = render_prompt(
            "prompts.citation_handlers.standard_citation_handler.standardcitationhandler.analyze_followup.fact_check_prompt",
            previous_knowledge=previous_knowledge,
            formatted_sources=formatted_sources,
        )
        if self.is_fact_checking_enabled():
            fact_check_response = self._invoke_text(fact_check_prompt)
        else:
            fact_check_response = ""

        current_timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )

        output_prefix = self._get_output_instruction_prefix()

        prompt = render_prompt(
            "prompts.citation_handlers.standard_citation_handler.standardcitationhandler.analyze_followup.prompt",
            output_prefix=output_prefix,
            previous_knowledge=previous_knowledge,
            question=question,
            formatted_sources=formatted_sources,
            current_timestamp=current_timestamp,
            fact_check_response=fact_check_response,
        )

        response = self._invoke_with_streaming(prompt)

        return {"content": response, "documents": documents}
