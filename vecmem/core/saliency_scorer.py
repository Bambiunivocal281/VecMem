"""
Saliency scoring for memory traces using the RIF model:
Recency, Importance, Frequency.

Score = 0.4 * recency + 0.4 * importance + 0.2 * frequency
"""

import time
import math
from typing import Dict
from threading import Lock

from .models import MemoryTrace


class SaliencyScorer:
    # Weight constants
    _RECENCY_WEIGHT = 0.4
    _IMPORTANCE_WEIGHT = 0.4
    _FREQUENCY_WEIGHT = 0.2
    _FREQUENCY_NORMALIZER = 10.0  # Frequency is capped at 10

    def __init__(self, half_life_seconds: int = 3600):
        """
        Creates a saliency scorer.

        Args:
            half_life_seconds: The half‑life for recency decay (in seconds).
                               Default is 1 hour (3600 seconds).
        """
        self._half_life_ms = half_life_seconds * 1000
        self._frequency_map: Dict[str, int] = {}
        self._lock = Lock()

    def record_retrieval(self, trace_id: str) -> None:
        """
        Records a retrieval hit for a trace (increases its frequency).

        The frequency counter is incremented each time the trace is returned
        by a `recall` operation. This affects the frequency component of the
        saliency score.

        Args:
            trace_id: The ID of the trace that was retrieved.
        """
        with self._lock:
            self._frequency_map[trace_id] = self._frequency_map.get(trace_id, 0) + 1

    def score(self, trace: MemoryTrace) -> float:
        """
        Computes the saliency score for a trace.

        The score is calculated as:
            Score = 0.4*recency + 0.4*importance + 0.2*normalized_frequency

        Each component is clamped to the range [0.0, 1.0].

        Args:
            trace: The memory trace to score.

        Returns:
            A float in [0.0, 1.0], where higher values mean more salient.
        """
        recency = self._compute_recency(trace.timestamp)
        importance = trace.metadata.get("importance", 0.5)
        frequency = self._compute_frequency(trace.id)

        # Clamp to [0,1] (safety)
        recency = max(0.0, min(1.0, recency))
        importance = max(0.0, min(1.0, importance))
        # Normalize frequency: cap at 10, then divide by 10
        frequency = min(1.0, frequency / self._FREQUENCY_NORMALIZER)

        # Weighted sum
        return (self._RECENCY_WEIGHT * recency +
                self._IMPORTANCE_WEIGHT * importance +
                self._FREQUENCY_WEIGHT * frequency)

    def _compute_recency(self, timestamp_ms: int) -> float:
        """
        Computes recency score using exponential decay.

        Formula: recency = 0.5 ^ (age / half_life)

        Args:
            timestamp_ms: Trace timestamp in milliseconds (Unix epoch).

        Returns:
            Recency score in [0.0, 1.0]. A brand‑new trace returns 1.0.
        """
        current_time_ms = int(time.time() * 1000)
        age_ms = current_time_ms - timestamp_ms

        if age_ms <= 0:
            return 1.0

        # Exponential decay: half‑life determines how fast it drops
        return math.pow(0.5, age_ms / self._half_life_ms)

    def _compute_frequency(self, trace_id: str) -> float:
        """
        Returns the raw frequency count for a trace.

        Args:
            trace_id: The trace ID.

        Returns:
            Number of times this trace has been retrieved (0 if never).
        """
        with self._lock:
            return float(self._frequency_map.get(trace_id, 0))