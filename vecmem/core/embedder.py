"""
Embedder interface and implementations.
"""

from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector (unit length)
        """
        pass
    
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings."""
        pass
