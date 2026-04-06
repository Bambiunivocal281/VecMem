"""
Short-Term Memory (STM) implementation using a pluggable VectorIndex.
"""

from typing import List, Dict
from collections import deque
from threading import Lock

from .interfaces import VectorIndex, ShortTermMemory, Embedder
from .models import MemoryTrace


class VolatileSTM(ShortTermMemory):
    """
    Volatile, high‑speed short‑term memory using a pluggable VectorIndex.

    Implements FIFO eviction when capacity is reached. All methods are thread‑safe.
    The VectorIndex can be any implementation (HNSW, brute‑force, FAISS, etc.).
    """

    def __init__(self, vector_index: VectorIndex, embedder: Embedder, max_size: int = 1000):
        """
        Initialize Short-Term Memory.

        Args:
            vector_index: The vector index implementation (e.g., HnswIndex, BruteForceIndex)
            embedder: Converts text to embedding vectors
            max_size: Maximum number of traces before oldest are evicted (FIFO)
        """
        self.vector_index = vector_index
        self.embedder = embedder
        self.max_size = max_size

        # Storage
        self.traces: Dict[str, MemoryTrace] = {}
        self.recent_queue: deque[str] = deque()

        # Thread safety
        self.lock = Lock()

    def add(self, trace: MemoryTrace) -> None:
        """
        Add a memory trace to STM.

        Args:
            trace: Memory trace to store
        """
        with self.lock:
            # Enforce size limit - FIFO eviction
            while len(self.recent_queue) >= self.max_size:
                oldest_id = self.recent_queue.pop()
                if oldest_id in self.traces:
                    del self.traces[oldest_id]
                    self.vector_index.remove(oldest_id)

            # Add to queue (most recent first)
            self.recent_queue.appendleft(trace.id)

            # Store trace
            self.traces[trace.id] = trace

            # Index embedding
            vector = self.embedder.embed(trace.content)
            self.vector_index.add(trace.id, vector)

    def search(self, query: str, k: int) -> List[MemoryTrace]:
        """
        Search for relevant traces using semantic similarity.

        Args:
            query: Natural language query
            k: Maximum number of results

        Returns:
            List of relevant memory traces (ordered by similarity)
        """
        # Embed query
        query_vec = self.embedder.embed(query)

        # Search index (returns IDs)
        result_ids = self.vector_index.search(query_vec, k)

        # Retrieve traces (thread‑safe read)
        with self.lock:
            return [self.traces[tid] for tid in result_ids if tid in self.traces]

    def recent(self, limit: int) -> List[MemoryTrace]:
        """
        Get the N most recent traces (ordered by recency, newest first).

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of recent traces
        """
        with self.lock:
            recent_ids = list(self.recent_queue)[:limit]
            return [self.traces[tid] for tid in recent_ids if tid in self.traces]

    def remove(self, trace_id: str) -> None:
        """
        Remove a trace from STM by ID.

        Args:
            trace_id: ID of trace to remove
        """
        with self.lock:
            if trace_id in self.traces:
                del self.traces[trace_id]

            if trace_id in self.recent_queue:
                self.recent_queue.remove(trace_id)

            self.vector_index.remove(trace_id)

    def get_count(self) -> int:
        """Return current number of traces in STM."""
        with self.lock:
            return len(self.traces)