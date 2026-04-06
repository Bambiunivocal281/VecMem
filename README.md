# VecMem
A resource-aware Python library for LLM agent memory. It implements a cognitive-inspired dual-tier system: a fast, HNSW-backed Short-Term Memory (STM) and a persistent, distilled Long-Term Memory (LTM).

This is based on my java implementation: [VectorMemory](https://github.com/Utilitron/VectorMemory)

Unlike standard vector databases, VectorMemory is built to be resource-aware, automatically triggering memory consolidation and distillation when system pressure (RAM/VRAM) is detected.

This project is currently in active development and is not yet available on Maven Central. Follow the instructions below to build it from source.

🚀 Key Features
Dual-Tier Architecture: Manage transient "working" context (STM) and permanent "learned" knowledge (LTM).

Saliency-Based Retention: Memories are scored based on a weighted formula of Recency, Importance, and Frequency.

Pluggable STM Vector Indexing: Easily plug in HNSW (via `hnswlib`), FAISS, or brute‑force indexes for fast similarity search via the `VectorIndex` interface.

Provider-Agnostic LLM Integration: Easily plug in LlamaFFM, Ollama, or cloud APIs for embedding and distillation.

Minimal Dependency Core: The core requires only `numpy`. Vector indexing, storage backends, and LLM integrations are defined as abstract interfaces.
