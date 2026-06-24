"""Pivot routes must load from the lab policy's named-mapping form.

Regression: _load_runtime_pivot_routes only read pivot.default_route / pivot.routes,
but runtime_policy.full-chain.json declares routes as named keys
(pivot: {"internal-bridge": {...}}), so no route loaded -> identity_context_probe
gave empty pivot_route_candidates AND the server-side pivot transport could not
resolve any route, blocking the whole internal-zone chain.
"""

from __future__ import annotations

import json

import pytest

from src.integrations.mcp_lab import tools


def _write_policy(tmp_path, pivot: dict) -> str:
    path = tmp_path / "runtime_policy.json"
    path.write_text(json.dumps({"policy_version": "v1", "adapter_policy": {"pivot": pivot}}), encoding="utf-8")
    return str(path)


def test_named_mapping_pivot_route_loads(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _write_policy(
        tmp_path,
        {
            "internal-bridge": {
                "route_id": "route::pivot-ssh::internal",
                "via_host": "10.20.0.50",
                "destination_cidr": "10.30.0.0/24",
                "protocol": "tcp",
                "transport": {"adapter": "ssh"},
            }
        },
    )
    monkeypatch.setenv("AEGRA_RUNTIME_POLICY_PATH", policy)

    routes = tools._load_runtime_pivot_routes()
    assert len(routes) == 1
    assert routes[0]["via_host"] == "10.20.0.50"

    candidates = tools._pivot_route_candidates()
    assert len(candidates) == 1
    assert candidates[0]["via_host"] == "10.20.0.50"
    assert candidates[0]["destination_cidr"] == "10.30.0.0/24"

    resolved = tools._resolve_pivot_route("route::pivot-ssh::internal")
    assert resolved is not None
    assert resolved["via_host"] == "10.20.0.50"


def test_named_key_used_as_route_id_when_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _write_policy(tmp_path, {"edge": {"via_host": "10.20.0.50", "destination_cidr": "10.30.0.0/24"}})
    monkeypatch.setenv("AEGRA_RUNTIME_POLICY_PATH", policy)
    routes = tools._load_runtime_pivot_routes()
    assert len(routes) == 1
    assert routes[0]["route_id"] == "edge"


def test_legacy_default_route_form_still_loads(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _write_policy(tmp_path, {"default_route": {"route_id": "r1", "via_host": "10.20.0.50"}})
    monkeypatch.setenv("AEGRA_RUNTIME_POLICY_PATH", policy)
    routes = tools._load_runtime_pivot_routes()
    assert [r["route_id"] for r in routes] == ["r1"]
