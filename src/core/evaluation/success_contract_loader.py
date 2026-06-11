"""Load SuccessContract from YAML files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.core.evaluation.models import ConditionBinding, SuccessContract


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]

        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        import json

        with path.open() as fh:
            return json.load(fh)


class SuccessContractLoader:
    """Load SuccessContract from a YAML file path."""

    def load(self, path: str | Path) -> SuccessContract:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(os.getcwd()) / resolved
        data = _yaml_load(resolved)
        return self._parse(data)

    def load_from_dict(self, data: dict[str, Any]) -> SuccessContract:
        return self._parse(dict(data))

    def _parse(self, data: dict[str, Any]) -> SuccessContract:
        raw_bindings: dict[str, Any] = data.pop("condition_bindings", {})
        bindings: dict[str, ConditionBinding] = {}
        for name, binding_data in raw_bindings.items():
            if isinstance(binding_data, dict):
                bindings[name] = ConditionBinding(**binding_data)
            else:
                raise ValueError(f"condition_binding '{name}' must be a dict")
        return SuccessContract(condition_bindings=bindings, **data)


_default_loader = SuccessContractLoader()


def load_contract(path: str | Path) -> SuccessContract:
    return _default_loader.load(path)


def contract_from_dict(data: dict[str, Any]) -> SuccessContract:
    return _default_loader.load_from_dict(data)
