"""
Standard question generation implementation.
"""

from datetime import datetime, UTC
from typing import List, Optional

from loguru import logger

from .base_question import BaseQuestionGenerator
from local_deep_research.prompts import render_prompt


class StandardQuestionGenerator(BaseQuestionGenerator):
    """Standard question generator."""

    def generate_questions(
        self,
        current_knowledge: str,
        query: str,
        questions_per_iteration: int = 2,
        questions_by_iteration: Optional[dict] = None,
    ) -> List[str]:
        """Generate follow-up questions based on current knowledge."""
        now = datetime.now(UTC)
        current_time = now.strftime("%Y-%m-%d")
        questions_by_iteration = questions_by_iteration or {}

        logger.info("Generating follow-up questions...")

        if questions_by_iteration:
            prompt = render_prompt(
                "prompts.advanced_search_system.questions.standard_question.standardquestiongenerator.generate_questions.prompt",
                questions_per_iteration=questions_per_iteration,
                query=query,
                current_time=current_time,
                questions_by_iteration=str(questions_by_iteration),
                current_knowledge=current_knowledge,
            )
        else:
            prompt = render_prompt(
                "prompts.advanced_search_system.questions.standard_question.standardquestiongenerator.generate_questions.prompt_2",
                current_time=current_time,
                questions_per_iteration=questions_per_iteration,
                query=query,
            )

        response = self.model.invoke(prompt)

        # Handle both string responses and responses with .content attribute
        response_text = ""
        if hasattr(response, "content"):
            response_text = response.content
        else:
            # Handle string responses
            response_text = str(response)

        questions = [
            q.replace("Q:", "").strip()
            for q in response_text.split("\n")
            if q.strip().startswith("Q:")
        ][:questions_per_iteration]

        logger.info(f"Generated {len(questions)} follow-up questions")

        return questions

    def generate_sub_questions(
        self, query: str, context: str = ""
    ) -> List[str]:
        """
        Generate sub-questions from a main query.

        Args:
            query: The main query to break down
            context: Additional context for question generation

        Returns:
            List[str]: List of generated sub-questions
        """
        prompt = render_prompt(
            "prompts.advanced_search_system.questions.standard_question.standardquestiongenerator.generate_sub_questions.prompt",
            query=query,
            context=context,
        )

        try:
            response = self.model.invoke(prompt)

            # Handle both string responses and responses with .content attribute
            content = ""
            if hasattr(response, "content"):
                content = response.content
            else:
                # Handle string responses
                content = str(response)

            # Parse sub-questions from the response
            sub_questions = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Extract sub-question from numbered or bulleted list
                    parts = (
                        line.split(".", 1)
                        if "." in line
                        else line.split(" ", 1)
                    )
                    if len(parts) > 1:
                        sub_question = parts[1].strip()
                        sub_questions.append(sub_question)

            # Limit to at most 5 sub-questions
            return sub_questions[:5]
        except Exception:
            logger.exception("Error generating sub-questions")
            return []
