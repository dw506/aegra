"""Helpers for keeping validation evidence compact and structured."""

from __future__ import annotations

from typing import Any


def normalize_validation_evidence(value: dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-safe evidence object with predictable top-level keys."""

    raw = dict(value or {})
    return {
        "signals": list(raw.get("signals", [])) if isinstance(raw.get("signals"), list) else [],
        "checked_urls": list(raw.get("checked_urls", [])) if isinstance(raw.get("checked_urls"), list) else [],
        "status_codes": dict(raw.get("status_codes", {})) if isinstance(raw.get("status_codes"), dict) else {},
        "notes": str(raw.get("notes", "")),
    }


__all__ = ["normalize_validation_evidence"]
