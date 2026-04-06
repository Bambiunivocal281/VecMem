"""
Utility for safely extracting typed values from metadata dictionaries.

Provides type-safe accessors with default values, matching the Java Metadata class.
"""

from typing import Dict, Any, TypeVar, Optional

T = TypeVar('T')


class Metadata:
    """Helper class for safely extracting typed values from metadata maps."""

    @staticmethod
    def get_string(metadata: Dict[str, Any], key: str, default: str = "") -> str:
        """
        Get a string value from metadata.

        Args:
            metadata: The metadata dictionary
            key: The key to look up
            default: Default value if key not found or value is not a string

        Returns:
            The string value or default
        """
        value = metadata.get(key)
        return value if isinstance(value, str) else default

    @staticmethod
    def get_int(metadata: Dict[str, Any], key: str, default: int = 0) -> int:
        """
        Get an integer value from metadata.

        Args:
            metadata: The metadata dictionary
            key: The key to look up
            default: Default value if key not found or value is not an int

        Returns:
            The integer value or default
        """
        value = metadata.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, (float, str)):
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        return default

    @staticmethod
    def get_float(metadata: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """
        Get a float value from metadata.

        Args:
            metadata: The metadata dictionary
            key: The key to look up
            default: Default value if key not found or value cannot be converted

        Returns:
            The float value or default
        """
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default

    @staticmethod
    def get_double(metadata: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """
        Alias for get_float to match Java naming (double == float in Python).

        Args:
            metadata: The metadata dictionary
            key: The key to look up
            default: Default value if key not found or value cannot be converted

        Returns:
            The float value or default
        """
        return Metadata.get_float(metadata, key, default)

    @staticmethod
    def get_bool(metadata: Dict[str, Any], key: str, default: bool = False) -> bool:
        """
        Get a boolean value from metadata.

        Args:
            metadata: The metadata dictionary
            key: The key to look up
            default: Default value if key not found or value is not a bool

        Returns:
            The boolean value or default
        """
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'y')
        if isinstance(value, (int, float)):
            return bool(value)
        return default