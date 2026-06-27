"""
Flexible BrowseComp question generator with less prescriptive instructions.

Inherits from BrowseComp but gives the LLM more freedom in search strategy.
"""

from typing import Dict, List

from .browsecomp_question import BrowseCompQuestionGenerator
from local_deep_research.prompts import render_prompt


class FlexibleBrowseCompQuestionGenerator(BrowseCompQuestionGenerator):
    """
    BrowseComp variant with simplified, less prescriptive prompts.

    Gives the LLM more freedom to explore different search strategies
    instead of strict entity-combination rules.
    """

    def _generate_progressive_searches(
        self,
        query: str,
        current_knowledge: str,
        entities: Dict[str, List[str]],
        questions_by_iteration: dict,
        results_by_iteration: dict,
        num_questions: int,
        iteration: int,
    ) -> List[str]:
        """Generate searches with more freedom and less rigid instructions."""

        # Check if recent searches are failing
        recent_iterations = list(range(max(1, iteration - 5), iteration))
        zero_count = sum(
            1 for i in recent_iterations if results_by_iteration.get(i, 1) == 0
        )
        searches_failing = zero_count >= 3

        # Simpler strategy hint
        if searches_failing:
            hint = "Recent searches returned 0 results. Try broader, simpler queries."
        else:
            hint = "Continue exploring to answer the query."

        # Much simpler prompt - less prescriptive
        prompt = render_prompt(
            "prompts.advanced_search_system.questions.flexible_browsecomp_question.flexiblebrowsecompquestiongenerator.generate_progressive_searches.prompt",
            num_questions=num_questions,
            query=query,
            hint=hint,
            join_8=", ".join(
                entities["names"]
                + entities["temporal"]
                + entities["descriptors"]
            ),
            format_previous_searches_10=self._format_previous_searches(
                questions_by_iteration, results_by_iteration
            ),
            conditional_12=current_knowledge[: self.knowledge_truncate_length]
            if self.knowledge_truncate_length
            else current_knowledge,
            num_questions_2=num_questions,
        )

        response = self.model.invoke(prompt)
        content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Extract searches
        searches = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and not line.endswith(":") and len(line) > 5:
                for prefix in [
                    "Q:",
                    "Search:",
                    "-",
                    "*",
                    "•",
                    "1.",
                    "2.",
                    "3.",
                    "4.",
                    "5.",
                ]:
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                if line:
                    searches.append(line)

        # If not enough, fall back to parent's logic
        if len(searches) < num_questions:
            parent_searches = super()._generate_progressive_searches(
                query,
                current_knowledge,
                entities,
                questions_by_iteration,
                results_by_iteration,
                num_questions - len(searches),
                iteration,
            )
            searches.extend(parent_searches)

        return searches[:num_questions]
