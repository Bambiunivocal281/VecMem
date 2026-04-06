"""
Strategy for distilling a set of memory traces into a single knowledge item.
"""

from typing import List, Protocol, Callable
import uuid
from .models import MemoryTrace, KnowledgeItem


class DistillationStrategy(Protocol):
    """
    Functional interface for distilling multiple memory traces into a knowledge item.

    Implementations should compress a list of low‑saliency traces (usually
    from short‑term memory) into a single, consolidated knowledge item
    for long‑term storage.
    """
    def distill(self, traces: List[MemoryTrace]) -> KnowledgeItem:
        ...


def concatenating() -> Callable[[List[MemoryTrace]], KnowledgeItem]:
    """
    A simple concatenation strategy (no LLM) – for testing.

    Returns:
        A strategy function that joins trace contents with newlines.
    """

    def distill_fn(traces: List[MemoryTrace]) -> KnowledgeItem:
        content = "\n---\n".join(trace.content for trace in traces)
        return KnowledgeItem(
            id=str(uuid.uuid4()),
            content=content,
            metadata={"distilled": True, "sourceCount": len(traces)},
            version=1
        )

    return distill_fn


# For convenience, also create a class-based implementation
class ConcatenatingDistillation:
    """Simple concatenation distillation strategy (no LLM)."""

    def distill(self, traces: List[MemoryTrace]) -> KnowledgeItem:
        """
        Distills traces by concatenating their content.

        Args:
            traces: List of memory traces to consolidate.

        Returns:
            A knowledge item with concatenated content.
        """
        content = "\n---\n".join(trace.content for trace in traces)
        return KnowledgeItem(
            id=str(uuid.uuid4()),
            content=content,
            metadata={"distilled": True, "sourceCount": len(traces)},
            version=1
        )