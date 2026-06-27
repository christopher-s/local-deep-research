"""
Evidence evaluator for assessing evidence quality and relevance.
"""

from typing import Dict

from langchain_core.language_models import BaseChatModel
from loguru import logger

from ..constraints.base_constraint import Constraint
from .base_evidence import Evidence, EvidenceType
from local_deep_research.prompts import render_prompt


class EvidenceEvaluator:
    """Evaluates evidence quality and relevance."""

    def __init__(self, model: BaseChatModel):
        """Initialize the evidence evaluator."""
        self.model = model
        self.source_reliability = {
            "official": 1.0,
            "research": 0.95,
            "news": 0.8,
            "community": 0.6,
            "inference": 0.5,
            "speculation": 0.3,
        }

    def extract_evidence(
        self, search_result: str, candidate: str, constraint: Constraint
    ) -> Evidence:
        """Extract evidence from search results for a specific constraint."""
        prompt = render_prompt(
            "prompts.advanced_search_system.evidence.evaluator.evidenceevaluator.extract_evidence.prompt",
            candidate=candidate,
            constraint_description=constraint.description,
            constraint_type_value=constraint.type.value,
            constraint_value=constraint.value,
            search_result_excerpt=search_result[:3000],
        )

        response = self.model.invoke(prompt)
        content = response.content

        # Parse response
        parsed = self._parse_evidence_response(content)

        # Create evidence object
        # Safely parse confidence value, handling potential errors
        confidence_str = parsed.get("confidence", "0.5")
        try:
            confidence = float(confidence_str)
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            logger.warning(
                f"Failed to parse confidence value: {confidence_str}"
            )
            confidence = 0.5

        evidence = Evidence(
            claim=parsed.get("claim", "No clear claim"),
            type=self._parse_evidence_type(parsed.get("type", "speculation")),
            source=parsed.get("source", "Unknown"),
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            raw_text=parsed.get("quote", ""),
            metadata={
                "candidate": candidate,
                "constraint_id": constraint.id,
                "constraint_type": constraint.type.value,
            },
        )

        # Adjust confidence based on how well it matches the constraint
        evidence.confidence *= self._assess_match_quality(evidence, constraint)

        return evidence

    def _parse_evidence_response(self, content: str) -> Dict[str, str]:
        """Parse the LLM response into evidence components."""
        import re

        parsed = {}

        for line in content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key in [
                    "claim",
                    "type",
                    "source",
                    "confidence",
                    "reasoning",
                    "quote",
                ]:
                    # Special handling for confidence to extract just the float value
                    if key == "confidence":
                        # Extract the first float from the value string
                        match = re.search(r"(\d*\.?\d+)", value)
                        if match:
                            parsed[key] = match.group(1)
                        else:
                            parsed[key] = value
                    else:
                        parsed[key] = value

        return parsed

    def _parse_evidence_type(self, type_str: str) -> EvidenceType:
        """Parse evidence type from string."""
        type_map = {
            "direct_statement": EvidenceType.DIRECT_STATEMENT,
            "official_record": EvidenceType.OFFICIAL_RECORD,
            "research_finding": EvidenceType.RESEARCH_FINDING,
            "news_report": EvidenceType.NEWS_REPORT,
            "statistical_data": EvidenceType.STATISTICAL_DATA,
            "inference": EvidenceType.INFERENCE,
            "correlation": EvidenceType.CORRELATION,
            "speculation": EvidenceType.SPECULATION,
        }
        return type_map.get(type_str.lower(), EvidenceType.SPECULATION)

    def _assess_match_quality(
        self, evidence: Evidence, constraint: Constraint
    ) -> float:
        """Assess how well the evidence matches the constraint."""
        # This is a simplified version - could be made more sophisticated
        if constraint.value.lower() in evidence.claim.lower():
            return 1.0
        if any(
            word in evidence.claim.lower()
            for word in constraint.value.lower().split()
        ):
            return 0.8
        return 0.6  # Partial match at best
