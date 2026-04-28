"""Custom exceptions for the knowledge graph module."""

from __future__ import annotations


class KnowledgeGraphError(Exception):
    """Base error for all knowledge graph failures."""


class DuplicateEntityError(KnowledgeGraphError):
    """Raised when a node or edge with the same ID already exists."""


class EntityNotFoundError(KnowledgeGraphError):
    """Raised when a requested node or edge does not exist."""


class ValidationConstraintError(KnowledgeGraphError):
    """Raised when a graph-level constraint is violated."""
