"""Enums used by the knowledge graph module."""

from __future__ import annotations

from enum import Enum


class EntityStatus(str, Enum):
    """Lifecycle status for nodes and edges."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    VALIDATED = "validated"
    STALE = "stale"
    REVOKED = "revoked"


class NodeType(str, Enum):
    """Supported node types in the knowledge graph."""

    HOST = "Host"
    SERVICE = "Service"
    IDENTITY = "Identity"
    CREDENTIAL = "Credential"
    SESSION = "Session"
    PRIVILEGE_STATE = "PrivilegeState"
    DATA_ASSET = "DataAsset"
    OBSERVATION = "Observation"
    EVIDENCE = "Evidence"
    FINDING = "Finding"
    NETWORK_ZONE = "NetworkZone"
    GOAL = "Goal"


class EdgeType(str, Enum):
    """Supported relationship types in the knowledge graph."""

    HOSTS = "HOSTS"
    BELONGS_TO_ZONE = "BELONGS_TO_ZONE"
    IDENTITY_PRESENT_ON = "IDENTITY_PRESENT_ON"
    IDENTITY_AVAILABLE_ON = "IDENTITY_AVAILABLE_ON"
    AUTHENTICATES_AS = "AUTHENTICATES_AS"
    REUSES_CREDENTIAL = "REUSES_CREDENTIAL"
    SESSION_FOR = "SESSION_FOR"
    SESSION_ON = "SESSION_ON"
    HAS_PRIVILEGE_STATE = "HAS_PRIVILEGE_STATE"
    PRIVILEGE_SOURCE = "PRIVILEGE_SOURCE"
    APPLIES_TO_HOST = "APPLIES_TO_HOST"
    CAN_REACH = "CAN_REACH"
    PIVOTS_TO = "PIVOTS_TO"
    OBSERVED_ON = "OBSERVED_ON"
    SUPPORTED_BY = "SUPPORTED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    RELATED_TO = "RELATED_TO"
    CONTAINS = "CONTAINS"
    TARGETS = "TARGETS"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"


class ChangeOperation(str, Enum):
    """Mutation type recorded in graph change events."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
