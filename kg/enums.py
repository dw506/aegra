"""Compatibility exports for KG enums under the root package."""

from src.core.models.kg_enums import ChangeOperation, EdgeType, EntityStatus, NodeType

__all__ = ["ChangeOperation", "EdgeType", "EntityStatus", "NodeType"]
