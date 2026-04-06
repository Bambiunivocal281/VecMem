"""
 LTM backed by a vector store (e.g., H2 with vector extension, or external DB).
"""

from typing import List, Optional, Dict, Any

from .interfaces import VectorStore, LongTermMemory, Embedder
from .models import KnowledgeItem

class PersistentLTM(LongTermMemory):
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        """
        Creates a persistent LTM.

        Args:
            vector_store: The underlying vector store (e.g., SQLiteVecStore, InMemoryVectorStore).
            embedder: Embedder for converting query strings and item content into vectors.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.cache: Dict[str, KnowledgeItem] = {}  # Simple dict – add lock if multi‑threaded

    def store(self, item: KnowledgeItem) -> None:
        # Generate embedding from the item's content
        vector = self.embedder.embed(item.content)

        # Prepare metadata: include content and version for reconstruction,
        metadata = dict(item.metadata)
        metadata["content"] = item.content
        metadata["version"] = item.version

        # Persist in the vector store
        self.vector_store.upsert(item.id, vector, metadata)

        # Update in‑memory cache
        self.cache[item.id] = item

    def search(self, query: str, k: int) -> List[KnowledgeItem]:
        # Embed the query
        query_vec = self.embedder.embed(query)

        # Get raw results from the vector store
        results = self.vector_store.search(query_vec, k)

        # Convert each result to a KnowledgeItem, using cache if available
        knowledge_items = []
        for r in results:
            # Try to get from cache first
            if r.id in self.cache:
                knowledge_items.append(self.cache[r.id])
            else:
                # Reconstruct from metadata stored in the vector store
                content = r.metadata.get("content", "")
                version = r.metadata.get("version", 1)

                # Clean internal fields from user‑facing metadata
                clean_metadata = {k: v for k, v in r.metadata.items()
                                  if k not in ("content", "version")}

                item = KnowledgeItem(
                    id=r.id,
                    content=content,
                    metadata=clean_metadata,
                    version=version
                )
                self.cache[r.id] = item
                knowledge_items.append(item)

        return knowledge_items

    def find_by_id(self, id: str) -> Optional[KnowledgeItem]:
        return self.cache.get(id)
