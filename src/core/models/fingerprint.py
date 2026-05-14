"""Fingerprint models normalized from recon output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ComponentFingerprint(BaseModel):
    """One detected software/component signal."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    product: str | None = None
    version: str | None = None
    cpe: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class TechnologyStackFingerprint(BaseModel):
    """Technology stack signals for one service."""

    model_config = ConfigDict(extra="forbid")

    components: list[ComponentFingerprint] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ServiceFingerprint(BaseModel):
    """Normalized service fingerprint extracted from recon results."""

    model_config = ConfigDict(extra="forbid")

    service_id: str = Field(min_length=1)
    host: str | None = None
    port: int | None = None
    protocol: str | None = None
    service_name: str | None = None
    product: str | None = None
    version: str | None = None
    http_title: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    body_signals: list[str] = Field(default_factory=list)
    cpe: list[str] = Field(default_factory=list)
    stack: TechnologyStackFingerprint = Field(default_factory=TechnologyStackFingerprint)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "recon"
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ComponentFingerprint",
    "ServiceFingerprint",
    "TechnologyStackFingerprint",
]
