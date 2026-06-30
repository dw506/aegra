"""Enums used by the knowledge graph module."""

from __future__ import annotations

from enum import Enum


class EntityStatus(str, Enum):
    #表示节点或边的生命周期状态

    OBSERVED = "observed"                    #已观察到
    INFERRED = "inferred"                    #推断中
    VALIDATED = "validated"                    #已验证
    STALE = "stale"                            #过时
    REVOKED = "revoked"                        #已撤销


class NodeType(str, Enum):
    # 定义节点类型

    HOST = "Host"
    SERVICE = "Service"
    CREDENTIAL = "Credential"
    SESSION = "Session"
    VULNERABILITY = "Vulnerability"
    OBSERVATION = "Observation"
    EVIDENCE = "Evidence"
    FINDING = "Finding"
    NETWORK_ZONE = "NetworkZone"
    GOAL = "Goal"
    # Extended types for profile-driven evaluation
    POST_ACCESS_OBSERVATION = "PostAccessObservation"
    GOAL_CHECK = "GoalCheck"
    GOAL_PROOF = "GoalProof"
    PIVOT_ROUTE = "PivotRoute"


class EdgeType(str, Enum):
    #定义边类型

    HOSTS = "HOSTS"
    BELONGS_TO_ZONE = "BELONGS_TO_ZONE"
    SESSION_ON = "SESSION_ON"
    APPLIES_TO_HOST = "APPLIES_TO_HOST"
    CAN_REACH = "CAN_REACH"
    PIVOTS_TO = "PIVOTS_TO"
    OBSERVED_ON = "OBSERVED_ON"
    SUPPORTED_BY = "SUPPORTED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    RELATED_TO = "RELATED_TO"
    CONTAINS = "CONTAINS"
    TARGETS = "TARGETS"


class ChangeOperation(str, Enum):
    #表示图数据发生变化时的操作类型

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
