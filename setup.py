"""
VectorMemory - Python Implementation
A two-tier memory system for AI agents
"""

from setuptools import setup, find_packages

setup(
    name="vecmem",
    version="1.0.0",
    description="Two-tier memory system for AI agents with zero‑dependency core",
    author="Erik Ashcraft",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",          # only mandatory dependency
    ],
    extras_require={
        # Optional backends
        "stm": ["hnswlib>=0.7.0"],
        "ltm": ["sqlite-vec>=0.1.0"],
        "monitoring": ["psutil>=5.9.0"],
        "embeddings": [
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.0.290",
        ],
        "all": [
            "hnswlib>=0.7.0",
            "sqlite-vec>=0.1.0",
            "psutil>=5.9.0",
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)