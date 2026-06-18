"""Load and cache OperationProfile from YAML files.

The loader is the only component that reads profile.yml from disk.
Profile contents are passed into all evaluation components as
OperationProfile objects, never as raw YAML strings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.core.evaluation.models import OperationProfile, ZoneBinding


def _yaml_load(path: Path) -> dict[str, Any]:
    """Load a YAML file, falling back to stdlib if PyYAML is unavailable."""
    try:
        import yaml  # type: ignore[import]

        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        import json

        # Fallback: try JSON (useful in minimal test environments)
        with path.open() as fh:
            return json.load(fh)


class ProfileLoader:
    """Load OperationProfile from a YAML file path.

    No environment-specific knowledge. All environment values come from
    the YAML file itself.
    """

    def load(self, path: str | Path) -> OperationProfile:
        """Load a profile from path, returning an OperationProfile."""
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(os.getcwd()) / resolved
        data = _yaml_load(resolved)
        return self._parse(data)

    def load_from_dict(self, data: dict[str, Any]) -> OperationProfile:
        """Build an OperationProfile from an already-parsed dict."""
        return self._parse(data)

    def _parse(self, data: dict[str, Any]) -> OperationProfile:
        zone_bindings_raw: dict[str, Any] = data.pop("zone_bindings", {})
        zone_bindings: dict[str, ZoneBinding] = {}
        for ref, zone_data in zone_bindings_raw.items():
            if isinstance(zone_data, dict):
                rest = {key: value for key, value in zone_data.items() if key != "name"}
                zone_bindings[ref] = ZoneBinding(name=zone_data.get("name", ref), **rest)
            else:
                zone_bindings[ref] = ZoneBinding(name=ref)
        return OperationProfile(zone_bindings=zone_bindings, **data)


_default_loader = ProfileLoader()


def load_profile(path: str | Path) -> OperationProfile:
    """Convenience wrapper around ProfileLoader.load()."""
    return _default_loader.load(path)


def profile_from_dict(data: dict[str, Any]) -> OperationProfile:
    """Convenience wrapper around ProfileLoader.load_from_dict()."""
    return _default_loader.load_from_dict(data)
