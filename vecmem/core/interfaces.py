"""Abstract base classes for dependency inversion."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import numpy as np
from .models import MemoryTrace, KnowledgeItem, SearchResult


class MemoryUnit(ABC):
    """Base protocol for all memory units."""
    id: str
    content: str
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """Low‑level vector database interface."""

    @abstractmethod
    def upsert(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert or update a vector with associated metadata."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """Return top‑k results (id, score, metadata)."""
        pass


class VectorIndex(ABC):
    """A vector index for approximate nearest neighbour search."""

    @abstractmethod
    def add(self, id: str, vector: np.ndarray) -> None:
        """Adds a vector to the index with the given ID."""
        pass

    @abstractmethod
    def remove(self, id: str) -> None:
        """Removes a vector from the index by its ID."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[str]:
        """Searches for the k nearest neighbours to the query vector."""
        pass


class ShortTermMemory(ABC):
    """Volatile, high‑speed working memory."""

    @abstractmethod
    def add(self, trace: MemoryTrace) -> None:
        """Add a memory trace (embedding handled internally)."""
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> List[MemoryTrace]:
        """Semantic search by query string."""
        pass

    @abstractmethod
    def recent(self, limit: int) -> List[MemoryTrace]:
        """Most recent traces, newest first."""
        pass

    @abstractmethod
    def remove(self, id: str) -> None:
        """Remove a trace by ID."""
        pass


class LongTermMemory(ABC):
    """Persistent, distilled memory."""

    @abstractmethod
    def store(self, item: KnowledgeItem) -> None:
        """Store a knowledge item (embedding handled internally)."""
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> List[KnowledgeItem]:
        """Semantic search by query string."""
        pass

    @abstractmethod
    def find_by_id(self, id: str) -> Optional[KnowledgeItem]:
        """Retrieve by ID."""
        pass


class Embedder(ABC):
    """Abstract text embedder."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return normalized embedding vector."""
        pass

    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensionality."""
        pass


class SaliencyScorer(ABC):
    """Abstract saliency scorer for consolidation decisions."""

    @abstractmethod
    def score(self, unit: MemoryUnit) -> float:
        """Return saliency score in [0,1] (higher = more salient)."""
        pass

    @abstractmethod
    def record_retrieval(self, unit_id: str) -> None:
        """Notify scorer that a unit was retrieved (affects frequency)."""
        pass