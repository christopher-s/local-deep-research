"""
Constraint analyzer for extracting constraints from queries.
"""

import re
from typing import List

from langchain_core.language_models import BaseChatModel
from loguru import logger

from .base_constraint import Constraint, ConstraintType
from local_deep_research.prompts import render_prompt


class ConstraintAnalyzer:
    """Analyzes queries to extract constraints."""

    def __init__(self, model: BaseChatModel):
        """Initialize the constraint analyzer."""
        self.model = model

    def extract_constraints(self, query: str) -> List[Constraint]:
        """Extract constraints from a query."""
        prompt = render_prompt(
            "prompts.advanced_search_system.constraints.constraint_analyzer.constraintanalyzer.extract_constraints.prompt",
            query=query,
        )

        response = self.model.invoke(prompt)
        content = response.content

        constraints = []
        current_constraint = {}
        constraint_id = 1

        for line in content.strip().split("\n"):
            line = line.strip()

            if line.startswith("CONSTRAINT_"):
                if current_constraint and all(
                    k in current_constraint
                    for k in ["type", "description", "value"]
                ):
                    constraint = Constraint(
                        id=f"c{constraint_id}",
                        type=self._parse_constraint_type(
                            current_constraint["type"]
                        ),
                        description=current_constraint["description"],
                        value=current_constraint["value"],
                        weight=self._parse_weight(
                            current_constraint.get("weight", 1.0)
                        ),
                    )
                    constraints.append(constraint)
                    constraint_id += 1
                current_constraint = {}
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ["type", "description", "value", "weight"]:
                    current_constraint[key] = value

        # Don't forget the last constraint
        if current_constraint and all(
            k in current_constraint for k in ["type", "description", "value"]
        ):
            constraint = Constraint(
                id=f"c{constraint_id}",
                type=self._parse_constraint_type(current_constraint["type"]),
                description=current_constraint["description"],
                value=current_constraint["value"],
                weight=self._parse_weight(
                    current_constraint.get("weight", 1.0)
                ),
            )
            constraints.append(constraint)

        logger.info(f"Extracted {len(constraints)} constraints from query")
        return constraints

    def _parse_constraint_type(self, type_str: str) -> ConstraintType:
        """Parse constraint type from string."""
        type_map = {
            "property": ConstraintType.PROPERTY,
            "name_pattern": ConstraintType.NAME_PATTERN,
            "event": ConstraintType.EVENT,
            "statistic": ConstraintType.STATISTIC,
            "temporal": ConstraintType.TEMPORAL,
            "location": ConstraintType.LOCATION,
            "comparison": ConstraintType.COMPARISON,
            "existence": ConstraintType.EXISTENCE,
        }
        return type_map.get(type_str.lower(), ConstraintType.PROPERTY)

    def _parse_weight(self, weight_value) -> float:
        """Parse weight value to float, handling text annotations.

        Args:
            weight_value: String or numeric weight value, possibly with text annotations

        Returns:
            float: Parsed weight value
        """
        if isinstance(weight_value, (int, float)):
            return float(weight_value)
        if isinstance(weight_value, str):
            # Extract the first number from the string
            match = re.search(r"(\d+(\.\d+)?)", weight_value)
            if match:
                return float(match.group(1))
        return 1.0  # Default weight
