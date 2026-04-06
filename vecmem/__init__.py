"""
VecMem: Two-tier memory system for AI agents.
"""

from .core.models import MemoryTrace, KnowledgeItem, SearchResult
from .core.interfaces import (
    MemoryUnit,
    VectorStore,
    VectorIndex,
    ShortTermMemory,
    LongTermMemory,
    Embedder,
    SaliencyScorer as SaliencyScorerInterface
)
from .core.memory_service import MemoryService
from .core.volatile_stm import VolatileSTM
from .core.persistent_ltm import PersistentLTM
from .core.saliency_scorer import SaliencyScorer
from .core.consolidator import Consolidator
from .core.distillation_strategy import (
    DistillationStrategy,
    concatenating
)
from .core.embedder import Embedder as EmbedderBase

__all__ = [
    # Models
    "MemoryTrace",
    "KnowledgeItem",
    "SearchResult",
    # Interfaces
    "MemoryUnit",
    "VectorStore",
    "VectorIndex",
    "ShortTermMemory",
    "LongTermMemory",
    "Embedder",
    "SaliencyScorerInterface",
    # Implementations
    "MemoryService",
    "VolatileSTM",
    "PersistentLTM",
    "SaliencyScorer",
    "Consolidator",
    "EmbedderBase",
    # Distillation
    "DistillationStrategy",
    "concatenating",
]