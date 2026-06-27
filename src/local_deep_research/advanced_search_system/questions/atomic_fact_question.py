"""
Atomic fact question generator for complex queries.
Decomposes complex queries into atomic, independently searchable facts.
"""

from loguru import logger
from typing import Dict, List, Optional

from .base_question import BaseQuestionGenerator
from local_deep_research.prompts import render_prompt


class AtomicFactQuestionGenerator(BaseQuestionGenerator):
    """
    Generates questions by decomposing complex queries into atomic facts.

    This approach prevents the system from searching for documents that match
    ALL criteria at once, instead finding facts independently and then reasoning
    about connections.
    """

    def generate_questions(
        self,
        current_knowledge: str,
        query: str,
        questions_per_iteration: int = 5,
        questions_by_iteration: Optional[Dict[int, List[str]]] = None,
    ) -> List[str]:
        """
        Generate atomic fact questions from a complex query.

        Args:
            current_knowledge: The accumulated knowledge so far
            query: The original research query
            questions_per_iteration: Number of questions to generate
            questions_by_iteration: Questions generated in previous iterations

        Returns:
            List of atomic fact questions
        """
        questions_by_iteration = questions_by_iteration or {}

        # On first iteration, decompose the query
        if not questions_by_iteration:
            return self._decompose_to_atomic_facts(query)

        # On subsequent iterations, fill knowledge gaps or explore connections
        return self._generate_gap_filling_questions(
            query,
            current_knowledge,
            questions_by_iteration,
            questions_per_iteration,
        )

    def _decompose_to_atomic_facts(self, query: str) -> List[str]:
        """Decompose complex query into atomic, searchable facts."""
        prompt = render_prompt(
            "prompts.advanced_search_system.questions.atomic_fact_question.atomicfactquestiongenerator.decompose_to_atomic_facts.prompt",
            query=query,
        )

        response = self.model.invoke(prompt)

        # Extract response text
        response_text = ""
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)

        # Parse questions
        questions = []
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 10:
                # Clean up any numbering or bullets
                for prefix in ["1.", "2.", "3.", "4.", "5.", "-", "*", "•"]:
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                questions.append(line)

        logger.info(f"Decomposed query into {len(questions)} atomic facts")
        return questions[:5]  # Limit to 5 atomic facts

    def _generate_gap_filling_questions(
        self,
        original_query: str,
        current_knowledge: str,
        questions_by_iteration: Dict[int, List[str]],
        questions_per_iteration: int,
    ) -> List[str]:
        """Generate questions to fill knowledge gaps or make connections."""

        # Check if we have enough information to start reasoning
        if len(questions_by_iteration) >= 3:
            prompt = render_prompt(
                "prompts.advanced_search_system.questions.atomic_fact_question.atomicfactquestiongenerator.generate_gap_filling_questions.prompt",
                original_query=original_query,
                current_knowledge=current_knowledge,
                format_previous_questions_6=self._format_previous_questions(
                    questions_by_iteration
                ),
                questions_per_iteration=questions_per_iteration,
            )
        else:
            # Still gathering basic facts
            prompt = render_prompt(
                "prompts.advanced_search_system.questions.atomic_fact_question.atomicfactquestiongenerator.generate_gap_filling_questions.prompt_2",
                original_query=original_query,
                format_previous_questions_4=self._format_previous_questions(
                    questions_by_iteration
                ),
                current_knowledge=current_knowledge,
                questions_per_iteration=questions_per_iteration,
            )

        response = self.model.invoke(prompt)

        # Extract response text
        response_text = ""
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)

        # Parse questions
        questions = []
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 10:
                # Clean up any numbering or bullets
                for prefix in ["1.", "2.", "3.", "4.", "5.", "-", "*", "•"]:
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                questions.append(line)

        return questions[:questions_per_iteration]
