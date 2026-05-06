"""In-memory TTL cache for market data and API responses."""

from __future__ import annotations

import time
from typing import Any


class DataCache:
    """Simple in-memory cache with TTL (time-to-live) expiration.

    Used to cache market data API responses to reduce API calls
    and respect rate limits.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
            max_size: Maximum number of entries before eviction.
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        """Get a value from the cache if it exists and hasn't expired.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if missing/expired.
        """
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache with the current timestamp.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        if len(self._cache) >= self._max_size:
            self._evict_expired()

        self._cache[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache.

        Args:
            key: Cache key to remove.
        """
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()

    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        now = time.time()
        expired_keys = [
            k for k, (_, ts) in self._cache.items() if now - ts > self._ttl
        ]
        for key in expired_keys:
            del self._cache[key]

    @property
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self._cache)
