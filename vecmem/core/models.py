"""
Core data models for RagLib VectorMemory.
"""

from typing import Dict, Any, Protocol
from dataclasses import dataclass, field


class MemoryUnit(Protocol):
    """Base protocol for all memory units (traces and knowledge items)."""

    id: str
    content: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class MemoryTrace:
    """
    A raw, volatile memory trace stored in short-term memory (STM).

    Traces are high-volume, noisy, and short-lived. They represent immediate
    events: user messages, code edits, debug logs, etc.
    """

    id: str
    content: str
    timestamp: int  # milliseconds since epoch
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Make metadata mutable by using default_factory
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', dict(self.metadata))


@dataclass(frozen=True)
class KnowledgeItem:
    """
    A consolidated, distilled knowledge item stored in long-term memory (LTM).

    Knowledge items are the result of the consolidation pipeline: they are
    compressed, curated, and stable. They represent high-density facts
    extracted from many memory traces.
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', dict(self.metadata))


@dataclass
class SearchResult:
    """Result from a vector search operation."""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
