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
    SESSION = "Session"
    EVIDENCE = "Evidence"
    FINDING = "Finding"
    NETWORK_ZONE = "NetworkZone"
    GOAL = "Goal"
    # Extended types for profile-driven evaluation
    GOAL_PROOF = "GoalProof"
    PIVOT_ROUTE = "PivotRoute"


class EdgeType(str, Enum):
    #定义边类型

    HOSTS = "HOSTS"
    BELONGS_TO_ZONE = "BELONGS_TO_ZONE"
    SUPPORTED_BY = "SUPPORTED_BY"
    TARGETS = "TARGETS"


class ChangeOperation(str, Enum):
    #表示图数据发生变化时的操作类型

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
