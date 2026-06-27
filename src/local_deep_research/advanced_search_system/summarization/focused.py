"""LLM summarizer biased toward a focus query."""

from langchain_core.language_models.chat_models import BaseChatModel

from local_deep_research.prompts import render_prompt

from .base import BaseSummarizer


class FocusedSummarizer(BaseSummarizer):
    """Produces a summary biased toward aspects relevant to a focus query."""

    def __init__(
        self,
        model: BaseChatModel,
        focus_query: str,
        max_sentences: int = 3,
        max_chars: int = 300,
    ):
        super().__init__(
            model, max_sentences=max_sentences, max_chars=max_chars
        )
        self.focus_query = focus_query

    def _build_prompt(self, content: str) -> str:
        return render_prompt(
            "prompts.advanced_search_system.summarization.focused",
            max_sentences=self.max_sentences,
            focus_query=repr(self.focus_query),
            content=content,
        )
