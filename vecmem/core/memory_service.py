"""
Main entry point for the VectorMemory system.

Manages both short‑term and long‑term memory, handles queries,
records retrieval frequency, and triggers consolidation under pressure.
"""

from typing import List, Dict, Any, Optional
import uuid
import time

from .models import MemoryTrace, KnowledgeItem, MemoryUnit
from .volatile_stm import HnswSTM  # concrete STM implementation
from .persistent_ltm import PersistentLTM    # concrete LTM implementation
from .saliency_scorer import SaliencyScorer
from .consolidator import Consolidator


class MemoryService:
    def __init__(
        self,
        stm: HnswSTM,               # or any ShortTermMemory implementation
        ltm: PersistentLTM,         # or any LongTermMemory implementation
        scorer: SaliencyScorer
    ):
        """
        Creates a memory service.

        Args:
            stm: Short‑term memory instance (must support `add`, `search`, `recent`, `remove`).
            ltm: Long‑term memory instance (must support `store`, `search`, `findById`).
            scorer: Saliency scorer for tracking retrieval frequency.
        """
        self.stm = stm
        self.ltm = ltm
        self.scorer = scorer

    def remember(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores a new memory trace (raw event) into STM.

        Args:
            content: The textual content of the memory.
            metadata: Optional metadata (e.g., importance, tags). Defaults to empty dict.

        Returns:
            The unique ID of the created trace.
        """
        if metadata is None:
            metadata = {}

        trace = MemoryTrace(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=int(time.time() * 1000),
            metadata=metadata.copy()   # copy to avoid external mutations
        )

        self.stm.add(trace)
        return trace.id

    def recall(self, query: str, k: int = 5) -> List[MemoryUnit]:
        """
        Recalls memories relevant to the query, merging results from STM and LTM.

        Args:
            query: Natural language query.
            k: Maximum total results to return.

        Returns:
            List of memory units (both `MemoryTrace` from STM and `KnowledgeItem` from LTM),
            ordered by relevance (most relevant first).
        """
        # Split k roughly equally between STM and LTM
        stm_k = k // 2
        ltm_k = k - stm_k

        # Search STM
        stm_results = self.stm.search(query, stm_k)

        # Record retrievals for saliency scoring (only for STM traces)
        for trace in stm_results:
            self.scorer.record_retrieval(trace.id)

        # Search LTM
        ltm_results = self.ltm.search(query, ltm_k)

        # Merge results (STM first, then LTM)
        merged: List[MemoryUnit] = []
        merged.extend(stm_results)
        merged.extend(ltm_results)

        return merged

    def tick(self, pressure: float, consolidator: Consolidator) -> None:
        """
        Triggers consolidation if system pressure is high.

        Args:
            pressure: Current resource pressure in [0.0, 1.0], where 1.0 means critical.
            consolidator: The consolidator instance that performs the actual consolidation.
        """
        # Only consolidate when pressure exceeds 85% (matching Java's 0.85)
        if pressure > 0.85:
            consolidator.consolidate()
