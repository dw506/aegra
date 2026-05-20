"""Operation lifecycle route registration.

Routes are currently registered by the compatibility facade in
``src.app.api``. This module marks the lifecycle route boundary for the next
step of the API split without changing public behavior.
"""

from __future__ import annotations

from typing import Any


def register_operation_routes(app: Any, orchestrator: Any, settings: Any) -> None:
    """Compatibility hook for operation lifecycle routes."""

    return None
