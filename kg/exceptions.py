"""Compatibility exports for KG exceptions under the root package."""

from src.core.models.kg_exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    KnowledgeGraphError,
    ValidationConstraintError,
)

__all__ = [
    "DuplicateEntityError",
    "EntityNotFoundError",
    "KnowledgeGraphError",
    "ValidationConstraintError",
]
