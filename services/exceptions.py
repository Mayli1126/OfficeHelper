class OfficeMateError(Exception):
    """Base exception for expected application failures."""


class RetrievalError(OfficeMateError):
    """Raised when the knowledge index cannot complete a query."""


class StorageError(OfficeMateError):
    """Raised when persisted metadata is unavailable or corrupted."""
