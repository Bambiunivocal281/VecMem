"""
Runs the consolidation process: moves low‑saliency traces from STM to LTM
after distilling them into a knowledge item.
"""

from typing import List, Callable
import uuid
from .interfaces import ShortTermMemory, LongTermMemory, Embedder, SaliencyScorer
from .distillation_strategy import DistillationStrategy


class Consolidator:
    def __init__(
        self,
        stm: ShortTermMemory,
        ltm: LongTermMemory,
        scorer: SaliencyScorer,
        distillation_strategy: DistillationStrategy,
        low_saliency_count: int = 50
    ):
        """
        Initialize the consolidator.

        Args:
            stm: Short‑term memory store (must support `recent()` and `delete()`).
            ltm: Long‑term memory store (must support `store(unit, vector)`).
            scorer: Saliency scorer for ranking traces.
            distillation_strategy: Strategy to distill multiple traces into one KnowledgeItem.
            low_saliency_count: Number of lowest‑saliency traces to consolidate each run.
        """
        self.stm = stm
        self.ltm = ltm
        self.scorer = scorer
        self.strategy = distillation_strategy
        self.low_saliency_count = low_saliency_count

    def consolidate(self) -> None:
        """
        Perform one consolidation cycle.
        """
        # Get recent traces (implementation‑specific; STM must provide this method)
        recent = self.stm.recent(1000)

        # Only consolidate if we have enough candidates to make it worthwhile
        if len(recent) < self.low_saliency_count:
            return

        # Sort by saliency ascending (lowest first) – these are the best candidates for eviction
        sorted_traces = sorted(recent, key=self.scorer.score)

        # Select the lowest‑saliency traces
        to_consolidate = sorted_traces[:self.low_saliency_count]

        if not to_consolidate:
            return

        # Distill the selected traces into a single knowledge item
        distilled = self.strategy.distill(to_consolidate)

        # Store the vector and knowledge item in LTM
        self.ltm.store(distilled)

        # Remove the original traces from STM
        for trace in to_consolidate:
            self.stm.remove(trace.id)