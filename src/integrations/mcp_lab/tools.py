"""Lab-only MCP tools used by the experimental LLM worker path."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import ssl
import subprocess
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from src.core.validation import ValidationPlan, ValidationResult, VulnerabilityProfile


DEFAULT_DISCOVERY_PATHS = ["/", "/robots.txt", "/sitemap.xml", "/admin", "/login"]
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_COMMAND_TIMEOUT_SECONDS = 120
MAX_COMMAND_TIMEOUT_SECONDS = 300
MAX_OUTPUT_CHARS = 20000
DEFAULT_FFUF_WORDS = ["admin", "login", "robots.txt", "sitemap.xml", "api", "debug", "health"]
SAFE_VALIDATION_PROFILES: dict[str, VulnerabilityProfile] = {
    "struts2-s2-045": VulnerabilityProfile(
        vulnerability_id="struts2-s2-045",
        cve="CVE-2017-5638",
        affected_products=["apache struts", "struts2", "struts2 showcase", "fileupload sample"],
        required_service="http",
        required_paths=["/"],
        safe_validation_methods=["http_probe", "validation_precheck"],
    ),
    "thinkphp-5-rce": VulnerabilityProfile(
        vulnerability_id="thinkphp-5-rce",
        affected_products=["thinkphp", "thinkphp 5", "thinkphp 5.0.23"],
        required_service="http",
        required_paths=["/"],
        safe_validation_methods=["http_probe", "validation_precheck"],
    ),
    "lab-http-accessible": VulnerabilityProfile(
        vulnerability_id="lab-http-accessible",
        affected_products=["generic-http"],
        required_service="http",
        required_paths=["/"],
        safe_validation_methods=["http_probe"],
    ),
    "lab-default-cred": VulnerabilityProfile(
        vulnerability_id="lab-default-cred",
        affected_products=["generic-http-basic"],
        required_service="http",
        required_paths=["/"],
        safe_validation_methods=["http_basic_auth_check"],
        requires_auth=True,
    ),
}
_HIDDEN_FIXTURE_CACHE: tuple[Path, float, dict[str, Any]] | None = None


LAB_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "run_command",
        "description": "Run one shell command in the isolated lab environment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "env": {"type": "object", "additionalProperties": {"type": "string"}},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "max_output_chars": {"type": "integer", "minimum": 100},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "nmap_scan",
        "description": "Run an nmap service discovery scan against one or more lab targets.",
        "inputSchema": {
            "type": "object",
            "required": ["target"],
            "properties": {
                "target": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                "ports": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                "service_detection": {"type": "boolean", "default": True},
                "skip_ping": {"type": "boolean", "default": True},
                "no_dns": {"type": "boolean", "default": True},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "http_probe",
        "description": "Probe one HTTP endpoint and return status, headers and a short body excerpt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "target": {"type": "string"},
                "scheme": {"type": "string", "default": "http"},
                "port": {"type": "integer"},
                "path": {"type": "string", "default": "/"},
                "method": {"type": "string", "default": "GET"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "web_fingerprint",
        "description": "Fetch a web page and extract title, server header and content type.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "target": {"type": "string"},
                "scheme": {"type": "string", "default": "http"},
                "port": {"type": "integer"},
                "path": {"type": "string", "default": "/"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "web_discover",
        "description": "Probe common paths on a lab web target.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base_url": {"type": "string"},
                "target": {"type": "string"},
                "scheme": {"type": "string", "default": "http"},
                "port": {"type": "integer"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "dns_lookup",
        "description": "Resolve hostnames in the lab environment.",
        "inputSchema": {
            "type": "object",
            "required": ["hostname"],
            "properties": {
                "hostname": {"type": "string"},
                "record_type": {"type": "string", "default": "A"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "tls_probe",
        "description": "Inspect a TLS endpoint certificate in the lab environment.",
        "inputSchema": {
            "type": "object",
            "required": ["host"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "default": 443},
                "server_name": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "tcp_connect_probe",
        "description": "Check whether one TCP host:port is reachable from the lab environment.",
        "inputSchema": {
            "type": "object",
            "required": ["host", "port"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "http_basic_auth_check",
        "description": "Validate one provided HTTP Basic credential against one endpoint; no brute forcing.",
        "inputSchema": {
            "type": "object",
            "required": ["url", "username", "password"],
            "properties": {
                "url": {"type": "string"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "vuln_profile_match",
        "description": "Match known bounded vulnerability validation profiles to observed lab service metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "product": {"type": "string"},
                "version": {"type": "string"},
                "target_url": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "validation_precheck",
        "description": "Check non-destructive preconditions for a safe validation profile against a lab target.",
        "inputSchema": {
            "type": "object",
            "required": ["target_url", "profile_id"],
            "properties": {
                "target_url": {"type": "string"},
                "profile_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "safe_vuln_validate",
        "description": "Run one bounded non-destructive vulnerability validation profile against an authorized lab target.",
        "inputSchema": {
            "type": "object",
            "required": ["target_url", "profile_id"],
            "properties": {
                "target_url": {"type": "string"},
                "profile_id": {"type": "string"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "safe_mode": {"type": "boolean", "default": True},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "credential_check",
        "description": "Validate one provided credential against one lab auth service; no brute forcing.",
        "inputSchema": {
            "type": "object",
            "required": ["auth_method", "target_url", "credential_id", "username", "password"],
            "properties": {
                "auth_method": {"type": "string"},
                "target_url": {"type": "string"},
                "credential_id": {"type": "string"},
                "target_service_id": {"type": "string"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "session_probe",
        "description": "Check whether a lab session handle has enough metadata to be reused.",
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string"},
                "bound_target": {"type": "string"},
                "bound_identity": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "session_open_lab",
        "description": "Register a bounded lab session handle for runtime coordination.",
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string"},
                "bound_target": {"type": "string"},
                "bound_identity": {"type": "string"},
                "lease_seconds": {"type": "integer", "minimum": 1},
                "reuse_policy": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "identity_context_probe",
        "description": "Return declared lab identity context facts for a session or host.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": True},
    },
    {
        "name": "privilege_context_probe",
        "description": "Return declared lab privilege context facts without running escalation payloads.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": True},
    },
    {
        "name": "pivot_route_probe",
        "description": "Verify a candidate lab pivot route with bounded TCP or HTTP reachability checks.",
        "inputSchema": {
            "type": "object",
            "required": ["destination_host", "destination_port"],
            "properties": {
                "route_id": {"type": "string"},
                "source_host": {"type": "string"},
                "via_host": {"type": "string"},
                "session_id": {"type": "string"},
                "destination_host": {"type": "string"},
                "destination_port": {"type": "integer"},
                "protocol": {"type": "string", "default": "tcp"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "internal_service_discover",
        "description": "Probe one internal lab service using bounded TCP or HTTP checks, optionally through a registered pivot route.",
        "inputSchema": {
            "type": "object",
            "required": ["host", "port"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "route_id": {"type": "string"},
                "session_id": {"type": "string"},
                "protocol": {"type": "string", "default": "tcp"},
                "path": {"type": "string", "default": "/"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "pivoted_nmap_scan",
        "description": "Run bounded service discovery from a configured pivot route against an authorized restricted-zone target or CIDR.",
        "inputSchema": {
            "type": "object",
            "required": ["target"],
            "properties": {
                "target": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                "ports": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                "route_id": {"type": "string"},
                "service_detection": {"type": "boolean", "default": True},
                "skip_ping": {"type": "boolean", "default": True},
                "no_dns": {"type": "boolean", "default": True},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "chain_goal_check",
        "description": "Alias for bounded goal condition checks used at the end of an attack-chain validation.",
        "inputSchema": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "expected_status": {"type": "integer"},
                "body_contains": {"type": "string"},
                "fixture_marker_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "goal_check",
        "description": "Check a bounded goal condition such as HTTP status or body substring.",
        "inputSchema": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "expected_status": {"type": "integer"},
                "body_contains": {"type": "string"},
                "fixture_marker_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "artifact_store",
        "description": "Store bounded evidence text or JSON as a local artifact and return an artifact reference.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "content": {"type": "string"},
                "json_content": {"type": "object"},
                "content_type": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "nuclei_scan",
        "description": "Run a bounded nuclei scan against one lab URL when nuclei is installed.",
        "inputSchema": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "templates": {"type": "array", "items": {"type": "string"}},
                "severity": {"type": "array", "items": {"type": "string"}},
                "rate_limit": {"type": "integer", "minimum": 1},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "whatweb_fingerprint",
        "description": "Run whatweb fingerprinting against one lab URL when whatweb is installed.",
        "inputSchema": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "ffuf_discover",
        "description": "Run a bounded ffuf path discovery against one lab base URL when ffuf is installed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base_url": {"type": "string"},
                "url": {"type": "string"},
                "wordlist": {"type": "string"},
                "words": {"type": "array", "items": {"type": "string"}},
                "match_status": {"type": "array", "items": {"type": "integer"}},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "lab_authorized_exploit_execute",
        "description": (
            "Execute a bounded, lab-authorized exploit via a pre-registered exploit profile. "
            "exploit_profile_id must reference a profile in configs/exploit_profiles/. "
            "Only operations declared in the profile's allowed_operations are executed. "
            "All evidence is written to /opt/aegra/ paths. Raw markers are never returned."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["exploit_profile_id", "target_url"],
            "properties": {
                "exploit_profile_id": {"type": "string"},
                "target_url": {"type": "string"},
                "session_id": {"type": "string"},
                "route_id": {"type": "string"},
                "extra_params": {"type": "object", "additionalProperties": {"type": "string"}},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "post_access_observe",
        "description": (
            "Observe post-access lab artifacts from declared drop zones (/opt/aegra/flags/, "
            "/opt/aegra/hints/, /opt/aegra/loot/). "
            "Returns file names and types only. Flag contents are returned as SHA-256 hashes. "
            "Never reads /etc/, /root/, or any system paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string", "enum": ["flags", "hints", "loot"], "default": "hints"},
                "session_id": {"type": "string"},
                "route_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "read_lab_marker",
        "description": (
            "Read a specific lab goal marker and return its HMAC hash for proof submission. "
            "The raw marker value is never returned. Use the returned submission_hash "
            "as proof_token in GoalAgent evidence."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["marker_id"],
            "properties": {
                "marker_id": {"type": "string"},
                "goal_id": {"type": "string"},
                "session_id": {"type": "string"},
                "route_id": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "pivot_route_register",
        "description": (
            "Register a validated pivot route for runtime coordination. "
            "Requires destination_host and destination_port to already be confirmed reachable "
            "via pivot_route_probe. Returns a route_id for use by internal tools."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["destination_host", "destination_port"],
            "properties": {
                "route_id": {"type": "string"},
                "source_host": {"type": "string"},
                "via_host": {"type": "string"},
                "session_id": {"type": "string"},
                "destination_host": {"type": "string"},
                "destination_port": {"type": "integer"},
                "protocol": {"type": "string", "default": "tcp"},
                "zone_ref": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "internal_goal_check",
        "description": (
            "Check a goal condition on an internal service reachable via a registered pivot route. "
            "Requires route_id or session_id. Returns goal_satisfied and opaque evidence_refs. "
            "Never returns raw marker or token values."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["host", "port"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "path": {"type": "string", "default": "/"},
                "route_id": {"type": "string"},
                "session_id": {"type": "string"},
                "expected_status": {"type": "integer"},
                "body_contains": {"type": "string"},
                "fixture_marker_id": {"type": "string"},
                "protocol": {"type": "string", "default": "http"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "controlled_data_read_proof",
        "description": (
            "Read a bounded, non-destructive proof from an authorized data service reachable "
            "through existing access or a pivot route. Connection secrets are resolved from "
            "lab configuration/environment and raw values are not returned."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["host", "port"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "service": {"type": "string"},
                "route_id": {"type": "string"},
                "session_id": {"type": "string"},
                "database": {"type": "string"},
                "username": {"type": "string"},
                "query": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "success_condition_check",
        "description": (
            "Return a redacted summary of the current success_condition_progress from runtime metadata. "
            "Reports eligible_for_stop, satisfied conditions, and missing conditions. "
            "Never exposes private rubric or raw proof values."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation_id": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },
]

OPTIONAL_TOOL_BINARIES: dict[str, str] = {
    "nuclei_scan": "nuclei",
    "whatweb_fingerprint": "whatweb",
    "ffuf_discover": "ffuf",
}


def lab_tool_specs(*, include_unavailable: bool = False) -> list[dict[str, Any]]:
    """Return tool specs that reflect binaries available in the current runtime."""

    specs: list[dict[str, Any]] = []
    for spec in LAB_TOOL_SPECS:
        name = str(spec.get("name") or "")
        binary = OPTIONAL_TOOL_BINARIES.get(name)
        if binary and shutil.which(binary) is None:
            if not include_unavailable:
                continue
            annotated = dict(spec)
            annotated["available"] = False
            annotated["unavailable_reason"] = f"{binary} is not installed or not on PATH"
            specs.append(annotated)
            continue
        annotated = dict(spec)
        annotated["available"] = True
        specs.append(annotated)
    return specs


def call_lab_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one lab tool and return a unified structured payload."""

    if not _lab_mode_enabled():
        return _ensure_raw_output_ref(
            arguments=arguments,
            payload=_payload(
                success=False,
                stderr="AEGRA_LAB_MODE=1 is required before lab MCP tools can execute",
                exit_code="lab_mode_required",
                parsed={"runtime_hints": {"required_env": "AEGRA_LAB_MODE=1"}},
            ),
        )

    try:
        if name == "run_command":
            return _ensure_raw_output_ref(arguments=arguments, payload=_run_command(arguments))
        if name == "nmap_scan":
            return _ensure_raw_output_ref(arguments=arguments, payload=_nmap_scan(arguments))
        if name == "http_probe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_http_probe(arguments))
        if name == "web_fingerprint":
            return _ensure_raw_output_ref(arguments=arguments, payload=_web_fingerprint(arguments))
        if name == "web_discover":
            return _ensure_raw_output_ref(arguments=arguments, payload=_web_discover(arguments))
        if name == "dns_lookup":
            return _ensure_raw_output_ref(arguments=arguments, payload=_dns_lookup(arguments))
        if name == "tls_probe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_tls_probe(arguments))
        if name == "tcp_connect_probe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_tcp_connect_probe(arguments))
        if name == "http_basic_auth_check":
            return _ensure_raw_output_ref(arguments=arguments, payload=_http_basic_auth_check(arguments))
        if name == "vuln_profile_match":
            return _ensure_raw_output_ref(arguments=arguments, payload=_vuln_profile_match(arguments))
        if name == "validation_precheck":
            return _ensure_raw_output_ref(arguments=arguments, payload=_validation_precheck(arguments))
        if name == "safe_vuln_validate":
            return _ensure_raw_output_ref(arguments=arguments, payload=_safe_vuln_validate(arguments))
        if name == "credential_check":
            return _ensure_raw_output_ref(arguments=arguments, payload=_credential_check(arguments))
        if name == "session_probe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_session_probe(arguments))
        if name == "session_open_lab":
            return _ensure_raw_output_ref(arguments=arguments, payload=_session_open_lab(arguments))
        if name == "identity_context_probe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_identity_context_probe(arguments))
        if name == "privilege_context_probe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_privilege_context_probe(arguments))
        if name == "pivot_route_probe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_pivot_route_probe(arguments))
        if name == "internal_service_discover":
            return _ensure_raw_output_ref(arguments=arguments, payload=_internal_service_discover(arguments))
        if name == "pivoted_nmap_scan":
            return _ensure_raw_output_ref(arguments=arguments, payload=_pivoted_nmap_scan(arguments))
        if name == "chain_goal_check":
            return _ensure_raw_output_ref(arguments=arguments, payload=_goal_check(arguments))
        if name == "goal_check":
            return _ensure_raw_output_ref(arguments=arguments, payload=_goal_check(arguments))
        if name == "artifact_store":
            return _ensure_raw_output_ref(arguments=arguments, payload=_artifact_store(arguments))
        if name == "nuclei_scan":
            return _ensure_raw_output_ref(arguments=arguments, payload=_nuclei_scan(arguments))
        if name == "whatweb_fingerprint":
            return _ensure_raw_output_ref(arguments=arguments, payload=_whatweb_fingerprint(arguments))
        if name == "ffuf_discover":
            return _ensure_raw_output_ref(arguments=arguments, payload=_ffuf_discover(arguments))
        if name == "lab_authorized_exploit_execute":
            return _ensure_raw_output_ref(arguments=arguments, payload=_lab_authorized_exploit_execute(arguments))
        if name == "post_access_observe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_post_access_observe(arguments))
        if name == "read_lab_marker":
            return _ensure_raw_output_ref(arguments=arguments, payload=_read_lab_marker(arguments))
        if name == "pivot_route_register":
            return _ensure_raw_output_ref(arguments=arguments, payload=_pivot_route_register(arguments))
        if name == "internal_goal_check":
            return _ensure_raw_output_ref(arguments=arguments, payload=_internal_goal_check(arguments))
        if name == "controlled_data_read_proof":
            return _ensure_raw_output_ref(arguments=arguments, payload=_controlled_data_read_proof(arguments))
        if name == "success_condition_check":
            return _ensure_raw_output_ref(arguments=arguments, payload=_success_condition_check(arguments))
    except Exception as exc:
        return _ensure_raw_output_ref(arguments=arguments, payload=_payload(success=False, stderr=str(exc), exit_code="tool_error"))
    return _ensure_raw_output_ref(arguments=arguments, payload=_payload(success=False, stderr=f"unknown lab MCP tool: {name}", exit_code="unknown_tool"))


def _run_command(arguments: dict[str, Any]) -> dict[str, Any]:
    timeout = max(1, min(_int(arguments.get("timeout_seconds"), DEFAULT_COMMAND_TIMEOUT_SECONDS), MAX_COMMAND_TIMEOUT_SECONDS))
    max_output = _int(arguments.get("max_output_chars"), MAX_OUTPUT_CHARS)
    cwd = arguments.get("cwd")
    cwd_text = str(cwd) if cwd else None
    if cwd_text and not Path(cwd_text).exists():
        return _payload(
            success=False,
            stderr=f"cwd does not exist: {cwd_text}",
            exit_code="cwd_error",
            parsed={"runtime_hints": {"blocked_by": "cwd_error", "cwd": cwd_text}},
        )
    env = os.environ.copy()
    env_keys: list[str] = []
    if isinstance(arguments.get("env"), dict):
        env_keys = sorted(str(key) for key in arguments["env"])
        env.update({str(key): str(value) for key, value in arguments["env"].items()})

    argv = arguments.get("argv")
    started = time.perf_counter()
    shell_command_used = False
    command_label = ""
    argv_list: list[str] | None = None
    if isinstance(argv, list) and argv:
        argv_list = [str(part) for part in argv]
        command_label = " ".join(argv_list)
        run_target: str | list[str] = argv_list
    else:
        command = str(arguments.get("command", "")).strip()
        if not command:
            return _payload(success=False, stderr="run_command requires command or argv", exit_code="missing_command")
        shell_command_used = True
        command_label = command
        run_target = command

    try:
        completed = subprocess.run(
            run_target,
            cwd=cwd_text,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell_command_used,
            encoding="utf-8",
            errors="replace",
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode: int | str = completed.returncode
        success = completed.returncode == 0
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        stdout = _decode_timeout_output(exc.stdout)
        stderr = _decode_timeout_output(exc.stderr) or f"command timed out after {timeout} seconds"
        returncode = "timeout"
        success = False
    except (OSError, subprocess.SubprocessError) as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        stdout = ""
        stderr = str(exc)
        returncode = "command_error"
        success = False

    raw_output_ref = _write_raw_tool_output(
        arguments=arguments,
        payload={
            "command": command_label,
            "argv": argv_list,
            "cwd": cwd_text,
            "timeout_seconds": timeout,
            "returncode": returncode,
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "shell_command_used": shell_command_used,
            "env_keys": env_keys,
        },
    )
    stdout_excerpt = _limit(stdout, max_output)
    stderr_excerpt = _limit(stderr, max_output)

    parsed = _default_parsed()
    parsed["writeback_hints"] = {
        "observation_category": "command_execution",
        "command": command_label,
        "raw_output_ref": raw_output_ref,
    }
    return _payload(
        success=success,
        stdout=stdout_excerpt,
        stderr=stderr_excerpt,
        exit_code=returncode,
        parsed=parsed,
        metadata={
            "command": command_label,
            "argv": argv_list,
            "cwd": cwd_text,
            "timeout_seconds": timeout,
            "returncode": returncode,
            "duration_ms": duration_ms,
            "stdout_excerpt": stdout_excerpt,
            "stderr_excerpt": stderr_excerpt,
            "raw_output_ref": raw_output_ref,
            "shell_command_used": shell_command_used,
            "env_keys": env_keys,
        },
    )


def _nmap_scan(arguments: dict[str, Any]) -> dict[str, Any]:
    target = arguments.get("target")
    if target is None or (isinstance(target, str) and not target.strip()):
        raise ValueError("target is required")
    targets = _target_list(target)
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    ports = arguments.get("ports")
    argv = ["nmap"]
    if bool(arguments.get("no_dns", True)):
        argv.append("-n")
    if bool(arguments.get("skip_ping", True)):
        argv.append("-Pn")
    if bool(arguments.get("service_detection", True)):
        argv.append("-sV")
    if ports:
        argv.extend(["-p", ",".join(str(port) for port in ports) if isinstance(ports, list) else str(ports)])
    argv.extend(targets)
    result = _run_command(_command_args_from_tool(arguments, argv=argv, timeout_seconds=timeout))
    parsed = _parse_nmap_output(",".join(targets), result.get("stdout", ""))
    result["parsed"] = parsed
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    if (
        "Failed to resolve" in stderr
        or "No targets were specified" in stderr
        or "No targets were specified" in stdout
        or "0 IP addresses" in stdout
        or "0 hosts up" in stdout
    ):
        result["success"] = False
        if result.get("exit_code") in (None, 0, "0"):
            result["exit_code"] = "no_targets_scanned"
        result.setdefault("parsed", {}).setdefault("runtime_hints", {})["blocked_by"] = "nmap_no_targets_scanned"
    return result


def _target_list(value: Any) -> list[str]:
    if isinstance(value, list):
        targets = [str(item).strip() for item in value if str(item).strip()]
    else:
        targets = [item.strip() for item in str(value).split(",") if item.strip()]
    if not targets:
        raise ValueError("target must contain at least one target")
    return targets


def _http_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    url = _url_from_arguments(arguments)
    method = str(arguments.get("method", "GET")).upper()
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    response = _open_url(url, method=method, timeout=timeout)
    parsed = _parsed_http(url=url, response=response)
    stdout = json.dumps(
        {
            "url": url,
            "status": response["status"],
            "headers": response["headers"],
            "body_excerpt": response["body_excerpt"],
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return _payload(success=True, stdout=stdout, exit_code=response["status"], parsed=parsed)


def _web_fingerprint(arguments: dict[str, Any]) -> dict[str, Any]:
    url = _url_from_arguments(arguments)
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    response = _open_url(url, method="GET", timeout=timeout)
    title = _extract_title(response["body_excerpt"])
    parsed = _parsed_http(url=url, response=response)
    parsed["entities"].append(
        {
            "type": "web_fingerprint",
            "url": url,
            "title": title,
            "server": response["headers"].get("server"),
            "content_type": response["headers"].get("content-type"),
        }
    )
    parsed["writeback_hints"] = {"observation_category": "web_fingerprint", "url": url}
    stdout = json.dumps(parsed["entities"][-1], ensure_ascii=True, sort_keys=True)
    return _payload(success=True, stdout=stdout, exit_code=response["status"], parsed=parsed)


def _web_discover(arguments: dict[str, Any]) -> dict[str, Any]:
    base_url = str(arguments.get("base_url") or _url_from_arguments(arguments)).rstrip("/") + "/"
    paths = arguments.get("paths") if isinstance(arguments.get("paths"), list) else DEFAULT_DISCOVERY_PATHS
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    parsed = _default_parsed()
    found: list[dict[str, Any]] = []
    for raw_path in paths:
        path = str(raw_path)
        url = urljoin(base_url, path.lstrip("/"))
        try:
            response = _open_url(url, method="GET", timeout=timeout)
        except Exception:
            continue
        if int(response["status"]) < 400:
            item = {
                "type": "web_path",
                "url": url,
                "path": "/" + path.lstrip("/"),
                "status": response["status"],
                "content_type": response["headers"].get("content-type"),
            }
            found.append(item)
            parsed["entities"].append(item)
    parsed["writeback_hints"] = {"observation_category": "web_discovery", "base_url": base_url}
    stdout = json.dumps(found, ensure_ascii=True, sort_keys=True)
    return _payload(success=True, stdout=stdout, exit_code=0, parsed=parsed)


def _dns_lookup(arguments: dict[str, Any]) -> dict[str, Any]:
    hostname = _required(arguments, "hostname")
    infos = socket.getaddrinfo(hostname, None)
    addresses = sorted({info[4][0] for info in infos})
    parsed = _default_parsed()
    for address in addresses:
        parsed["entities"].append({"type": "dns_record", "hostname": hostname, "address": address})
    parsed["writeback_hints"] = {"observation_category": "dns_lookup", "hostname": hostname}
    return _payload(success=True, stdout=json.dumps(addresses, ensure_ascii=True), exit_code=0, parsed=parsed)


def _tls_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "host")
    port = _int(arguments.get("port"), 443)
    server_name = str(arguments.get("server_name") or host)
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    context = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=server_name) as tls_sock:
            cert = tls_sock.getpeercert()
            cipher = tls_sock.cipher()
            version = tls_sock.version()
    entity = {
        "type": "tls_certificate",
        "host": host,
        "port": port,
        "subject": cert.get("subject"),
        "issuer": cert.get("issuer"),
        "not_before": cert.get("notBefore"),
        "not_after": cert.get("notAfter"),
        "cipher": cipher,
        "version": version,
    }
    parsed = _default_parsed()
    parsed["entities"].append(entity)
    parsed["writeback_hints"] = {"observation_category": "tls_probe", "host": host, "port": port}
    return _payload(success=True, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), exit_code=0, parsed=parsed)


def _tcp_connect_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "host")
    port = _int(arguments.get("port"), 0)
    if port <= 0 or port > 65535:
        raise ValueError("port must be between 1 and 65535")
    timeout = _int(arguments.get("timeout_seconds"), 5)
    parsed = _default_parsed()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            reachable = True
            error = ""
    except OSError as exc:
        reachable = False
        error = str(exc)
    entity = {"type": "tcp_endpoint", "host": host, "port": port, "reachable": reachable}
    parsed["entities"].append(entity)
    parsed["runtime_hints"] = {"reachable": reachable, "failure_reason": error}
    parsed["writeback_hints"] = {"observation_category": "reachability", "host": host, "port": port}
    return _payload(
        success=reachable,
        stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True),
        stderr=error,
        exit_code=0 if reachable else "unreachable",
        parsed=parsed,
    )


def _http_basic_auth_check(arguments: dict[str, Any]) -> dict[str, Any]:
    import base64

    url = _required(arguments, "url")
    username = _required(arguments, "username")
    password = _required(arguments, "password")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    response = _open_url(url, method="GET", timeout=timeout, headers={"Authorization": f"Basic {token}"})
    authenticated = int(response["status"]) not in {401, 403}
    parsed = _parsed_http(url=url, response=response)
    finding = {
        "kind": "credential_validation",
        "url": url,
        "username": username,
        "authenticated": authenticated,
        "status": response["status"],
    }
    parsed["findings"].append(finding)
    credential_id = _string(arguments.get("credential_id"))
    target_service_id = _string(arguments.get("target_service_id")) or url
    if credential_id is not None:
        parsed["runtime_hints"] = {
            "credential_id": credential_id,
            "credential_status": "valid" if authenticated else "invalid",
            "bind_target": target_service_id if authenticated else None,
            "target_service_id": target_service_id,
            "principal": username,
            "reason": None if authenticated else f"http_status_{response['status']}",
        }
    parsed["writeback_hints"] = {"observation_category": "credential_validation", "url": url, "username": username}
    return _payload(
        success=authenticated,
        stdout=json.dumps(finding, ensure_ascii=True, sort_keys=True),
        exit_code=response["status"],
        parsed=parsed,
    )


def _vuln_profile_match(arguments: dict[str, Any]) -> dict[str, Any]:
    service = _string(arguments.get("service"))
    product = (_string(arguments.get("product")) or "").lower()
    match_text = _profile_match_text(arguments)
    parsed = _default_parsed()
    matches: list[dict[str, Any]] = []
    for profile in SAFE_VALIDATION_PROFILES.values():
        if service and profile.required_service and service.lower() != profile.required_service.lower():
            continue
        if profile.affected_products and not _profile_matches_signals(profile, service=service, product=product, match_text=match_text):
            continue
        item = profile.model_dump(mode="json")
        matches.append(item)
        candidate_id = f"vuln-candidate::{profile.vulnerability_id}::{_string(arguments.get('target_url')) or service or 'unknown'}"
        parsed["entities"].append(
            {
                "type": "VulnerabilityCandidate",
                "candidate_id": candidate_id,
                "vulnerability_id": profile.vulnerability_id,
                "matched_profile_id": profile.vulnerability_id,
                "affected_products": list(profile.affected_products),
                "target_url": _string(arguments.get("target_url")),
                "confidence": 0.75,
            }
        )
        parsed["findings"].append(
            {
                "kind": "VulnerabilityCandidate",
                "candidate_id": candidate_id,
                "matched_profile_id": profile.vulnerability_id,
                "evidence_refs": [],
                "confidence": 0.75,
            }
        )
    parsed["writeback_hints"] = {"observation_category": "vulnerability_profile_match"}
    return _payload(success=True, stdout=json.dumps(matches, ensure_ascii=True, sort_keys=True), parsed=parsed)


def _profile_match_text(arguments: dict[str, Any]) -> str:
    values: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for nested in value.values():
                add(nested)
            return
        if isinstance(value, list):
            for nested in value:
                add(nested)
            return
        text = str(value).strip()
        if text:
            values.append(text)

    for key in (
        "title",
        "path",
        "paths",
        "url",
        "target_url",
        "banner",
        "server",
        "headers",
        "framework",
        "fingerprint",
        "fingerprints",
        "body_excerpt",
        "content",
        "evidence",
        "observations",
    ):
        add(arguments.get(key))
    return " ".join(values).lower()


def _profile_matches_signals(profile: VulnerabilityProfile, *, service: str | None, product: str, match_text: str) -> bool:
    product_tokens = [item.lower() for item in profile.affected_products if item]
    if not product_tokens:
        return True
    haystack = " ".join([product, match_text]).lower()
    for token in product_tokens:
        if not token:
            continue
        if token.startswith("generic-http") and str(service or "").lower() in {"http", "https"}:
            return True
        if token in haystack:
            return True
        compact_token = re.sub(r"[^a-z0-9]+", "", token)
        compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
        if compact_token and compact_token in compact_haystack:
            return True
    return False


def _validation_precheck(arguments: dict[str, Any]) -> dict[str, Any]:
    target_url = _required(arguments, "target_url")
    profile = _profile(_required(arguments, "profile_id"))
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    checks = _run_profile_prechecks(target_url=target_url, profile=profile, timeout=timeout)
    passed = all(item["passed"] for item in checks)
    parsed = _default_parsed()
    plan = ValidationPlan(
        profile_id=profile.vulnerability_id,
        target_ref=target_url,
        preconditions=[f"path:{path}" for path in profile.required_paths],
        tool_sequence=[{"tool": method, "target_url": target_url} for method in profile.safe_validation_methods],
        expected_evidence=["reachable_required_paths"],
        timeout_seconds=timeout,
    )
    parsed["entities"].append({"type": "validation_plan", **plan.model_dump(mode="json")})
    parsed["runtime_hints"] = {"validation_precheck_passed": passed, "profile_id": profile.vulnerability_id}
    parsed["writeback_hints"] = {"observation_category": "validation_precheck", "url": target_url}
    return _payload(success=passed, stdout=json.dumps({"checks": checks}, ensure_ascii=True), exit_code=0 if passed else "precheck_failed", parsed=parsed)


def _safe_vuln_validate(arguments: dict[str, Any]) -> dict[str, Any]:
    if not bool(arguments.get("safe_mode", True)):
        return _payload(
            success=False,
            stderr="safe_vuln_validate only supports safe_mode=true",
            exit_code="unsafe_mode_rejected",
            parsed={"runtime_hints": {"blocked_by": "unsafe_mode_rejected"}},
        )
    target_url = _required(arguments, "target_url")
    profile = _profile(_required(arguments, "profile_id"))
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    checks = _run_profile_prechecks(target_url=target_url, profile=profile, timeout=timeout)
    precheck_passed = all(item["passed"] for item in checks)
    authenticated: bool | None = None
    if profile.requires_auth:
        username = _string(arguments.get("username"))
        password = _string(arguments.get("password"))
        if username is None or password is None:
            validation = ValidationResult(
                vulnerability_id=profile.vulnerability_id,
                status="blocked",
                confidence=0.0,
                safe_payload_summary="validation not executed because provided credential was missing",
                evidence={"checks": checks},
                tool={"name": "safe_vuln_validate"},
                failure_reason="credential_required",
            )
            return _validation_payload(validation=validation, target_url=target_url, success=False)
        auth_result = _http_basic_auth_check({"url": target_url, "username": username, "password": password, "timeout_seconds": timeout})
        authenticated = bool(auth_result.get("success"))
    status = "validated" if (precheck_passed and (authenticated is True or not profile.requires_auth)) else "not_detected"
    confidence = 0.92 if status == "validated" else 0.35
    validation = ValidationResult(
        vulnerability_id=profile.vulnerability_id,
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        safe_payload_summary="executed bounded profile checks only; no exploit payload executed",
        evidence={"checks": checks, "authenticated": authenticated},
        tool={"name": "safe_vuln_validate", "profile_id": profile.vulnerability_id},
    )
    return _validation_payload(validation=validation, target_url=target_url, success=status == "validated")


def _credential_check(arguments: dict[str, Any]) -> dict[str, Any]:
    auth_method = _required(arguments, "auth_method")
    if auth_method != "http_basic":
        return _payload(
            success=False,
            stderr=f"unsupported credential auth_method: {auth_method}",
            exit_code="unsupported_auth_method",
            parsed={"runtime_hints": {"blocked_by": "unsupported_auth_method", "auth_method": auth_method}},
        )
    return _http_basic_auth_check(
        {
            "url": _required(arguments, "target_url"),
            "username": _required(arguments, "username"),
            "password": _required(arguments, "password"),
            "credential_id": _required(arguments, "credential_id"),
            "target_service_id": arguments.get("target_service_id"),
            "timeout_seconds": arguments.get("timeout_seconds"),
        }
    )


def _session_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    session_id = _required(arguments, "session_id")
    reusable = bool(arguments.get("bound_target") or arguments.get("bound_identity"))
    parsed = _default_parsed()
    entity = {
        "type": "session",
        "session_id": session_id,
        "bound_target": _string(arguments.get("bound_target")),
        "bound_identity": _string(arguments.get("bound_identity")),
        "reusable": reusable,
    }
    parsed["entities"].append(entity)
    parsed["runtime_hints"] = {"session_id": session_id, "session_reusable": reusable}
    parsed["writeback_hints"] = {"observation_category": "session_probe"}
    return _payload(success=True, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), parsed=parsed)


def _session_open_lab(arguments: dict[str, Any]) -> dict[str, Any]:
    session_id = _required(arguments, "session_id")
    lease_seconds = _int(arguments.get("lease_seconds"), 300)
    parsed = _default_parsed()
    parsed["runtime_hints"] = {
        "open_session": True,
        "session_id": session_id,
        "bound_target": _string(arguments.get("bound_target")),
        "bound_identity": _string(arguments.get("bound_identity")),
        "lease_seconds": lease_seconds,
        "reuse_policy": _string(arguments.get("reuse_policy")) or "exclusive",
    }
    parsed["writeback_hints"] = {"observation_category": "session_open"}
    return _payload(success=True, stdout=json.dumps(parsed["runtime_hints"], ensure_ascii=True, sort_keys=True), parsed=parsed)


def _identity_context_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    identity = _string(arguments.get("identity")) or _string(arguments.get("username")) or "unknown"
    host = _string(arguments.get("host")) or _string(arguments.get("target")) or "unknown-host"
    pivot_candidates = _configured_pivot_route_candidates()
    parsed = _default_parsed()
    entity = {"type": "identity_context", "host": host, "identity": identity, "session_id": _string(arguments.get("session_id"))}
    parsed["entities"].append(entity)
    for candidate in pivot_candidates:
        parsed["entities"].append({"type": "PivotRouteCandidate", **candidate})
    parsed["runtime_hints"] = {
        "identity": identity,
        "host": host,
        "identity_context_observed": True,
        "pivot_route_candidates": pivot_candidates,
    }
    parsed["writeback_hints"] = {"observation_category": "identity_context"}
    return _payload(success=True, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), parsed=parsed)


def _privilege_context_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    identity = _string(arguments.get("identity")) or "unknown"
    host = _string(arguments.get("host")) or "unknown-host"
    privilege_level = _string(arguments.get("privilege_level")) or "low"
    groups = _string_items(arguments.get("groups"))
    parsed = _default_parsed()
    entity = {
        "type": "privilege_context",
        "host": host,
        "identity": identity,
        "groups": groups,
        "privilege_level": privilege_level,
        "can_write_paths": _string_items(arguments.get("can_write_paths")),
        "can_execute_commands": bool(arguments.get("can_execute_commands", False)),
        "sudo_available": bool(arguments.get("sudo_available", False)),
        "container_runtime": bool(arguments.get("container_runtime", False)),
    }
    parsed["entities"].append(entity)
    parsed["findings"].append({"kind": "privilege_precheck", "status": privilege_level, "confidence": 0.8})
    parsed["runtime_hints"] = {"privilege_level": privilege_level, "needs_privilege_task": privilege_level in {"low", "limited"}}
    parsed["writeback_hints"] = {"observation_category": "privilege_context"}
    return _payload(success=True, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), parsed=parsed)


def _pivot_route_probe(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "destination_host")
    port = _int(arguments.get("destination_port"), 0)
    if port <= 0 or port > 65535:
        raise ValueError("destination_port must be between 1 and 65535")
    protocol = (_string(arguments.get("protocol")) or "tcp").lower()
    timeout = _int(arguments.get("timeout_seconds"), 5)
    if protocol in {"http", "https"}:
        url = f"{protocol}://{host}:{port}/"
        try:
            response = _open_url(url, method="GET", timeout=timeout)
            reachable = int(response["status"]) < 500
            error = ""
        except Exception as exc:
            reachable = False
            error = str(exc)
    else:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                reachable = True
                error = ""
        except OSError as exc:
            pivot_probe = _probe_tcp_via_configured_pivot(host=host, port=port, timeout_seconds=timeout)
            reachable = pivot_probe["reachable"]
            error = "" if reachable else str(exc)
    route_id = _string(arguments.get("route_id")) or f"route::{_string(arguments.get('source_host')) or 'unknown-source'}::{host}:{port}"
    route_candidate = _configured_pivot_route_by_id(route_id)
    route_transport_validated = False
    destination_reachable = reachable
    if not reachable and route_candidate and route_candidate.get("destination_cidr") and _validate_configured_pivot_transport(timeout_seconds=timeout):
        reachable = True
        route_transport_validated = True
        error = ""
    parsed = _default_parsed()
    entity = {
        "type": "PivotRoute",
        "route_id": route_id,
        "source_host": _string(arguments.get("source_host")),
        "via_host": _string(arguments.get("via_host")),
        "destination_host": host,
        "destination_cidr": route_candidate.get("destination_cidr") if route_candidate else None,
        "port": port,
        "protocol": protocol,
        "status": "validated" if reachable else "unreachable",
        "destination_reachable": destination_reachable,
        "route_transport_validated": route_transport_validated,
        "confidence": 0.85 if reachable else 0.25,
    }
    parsed["entities"].append(entity)
    parsed["evidence"].append(
        {
            "kind": "reachability evidence",
            "route_id": route_id,
            "destination_host": host,
            "destination_cidr": route_candidate.get("destination_cidr") if route_candidate else None,
            "port": port,
            "reachable": reachable,
            "destination_reachable": destination_reachable,
            "route_transport_validated": route_transport_validated,
        }
    )
    parsed["runtime_hints"] = {
        "register_pivot_route": True,
        "route_id": route_id,
        "destination_host": host,
        "destination_cidr": route_candidate.get("destination_cidr") if route_candidate else None,
        "source_host": _string(arguments.get("source_host")),
        "via_host": _string(arguments.get("via_host")),
        "session_id": _string(arguments.get("session_id")),
        "allowed_ports": [port],
        "protocol": protocol,
        "reachable": reachable,
        "validated": reachable,
        "route_transport_validated": route_transport_validated,
        "reason": error,
    }
    parsed["writeback_hints"] = {"observation_category": "pivot_route_validation"}
    return _payload(success=reachable, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), stderr=error, exit_code=0 if reachable else "unreachable", parsed=parsed)


def _internal_service_discover(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "host")
    port = _int(arguments.get("port"), 0)
    protocol = (_string(arguments.get("protocol")) or "tcp").lower()
    route_id = _string(arguments.get("route_id"))
    session_id = _string(arguments.get("session_id"))
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    if protocol in {"http", "https"}:
        result = _http_probe(
            {
                "target": host,
                "port": port,
                "scheme": protocol,
                "path": arguments.get("path", "/"),
                "timeout_seconds": arguments.get("timeout_seconds"),
            }
        )
    else:
        result = _tcp_connect_probe({"host": host, "port": port, "timeout_seconds": timeout})
        if not result.get("success") and (route_id or session_id or _pivot_transport_available()):
            pivot_probe = _probe_tcp_via_configured_pivot(host=host, port=port, timeout_seconds=timeout)
            if pivot_probe["reachable"]:
                result = _payload(
                    success=True,
                    stdout=json.dumps({"host": host, "port": port, "reachable": True, "via_pivot": True}, sort_keys=True),
                    parsed=_default_parsed(),
                )
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else _default_parsed()
    service_name = _classify_service_name(port=port, protocol=protocol, hinted_service=_string(arguments.get("service")))
    reachable = bool(result.get("success"))
    evidence_id = f"internal-service::{route_id or session_id or 'configured-pivot'}::{host}:{port}"
    parsed["entities"].append(
        {
            "type": "InternalService",
            "host": host,
            "port": port,
            "protocol": protocol,
            "service_name": service_name,
            "via_pivot_route": route_id,
            "confidence": 0.75 if reachable else 0.25,
        }
    )
    parsed["hosts"] = [{"address": host, "zone_ref": "restricted"}] if reachable else []
    parsed["services"] = (
        [{"host": host, "address": host, "port": port, "protocol": protocol, "service_name": service_name, "zone_ref": "restricted"}]
        if reachable
        else []
    )
    parsed["evidence"].append({"kind": "reachability evidence", "evidence_id": evidence_id, "host": host, "port": port, "via_pivot_route": route_id})
    if reachable:
        conditions = ["internal_service_discovered_after_authorized_route", "restricted_zone_service_discovered"]
        if _is_data_service(port=port, service_name=service_name):
            conditions.append("restricted_data_service_discovered")
        parsed["runtime_hints"] = {
            **(parsed.get("runtime_hints") if isinstance(parsed.get("runtime_hints"), dict) else {}),
            "route_id": route_id,
            "session_id": session_id,
            "satisfied_conditions": [{"condition": name, "evidence_ids": [evidence_id]} for name in conditions],
        }
    result["parsed"] = parsed
    return result


def _pivoted_nmap_scan(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments.get("target") is None:
        raise ValueError("target is required")
    targets = _target_list(arguments.get("target"))
    timeout = max(1, min(_int(arguments.get("timeout_seconds"), DEFAULT_COMMAND_TIMEOUT_SECONDS), MAX_COMMAND_TIMEOUT_SECONDS))
    ports = arguments.get("ports")
    argv = ["nmap"]
    if bool(arguments.get("no_dns", True)):
        argv.append("-n")
    if bool(arguments.get("skip_ping", True)):
        argv.append("-Pn")
    if bool(arguments.get("service_detection", True)):
        argv.append("-sV")
    if ports:
        argv.extend(["-p", ",".join(str(port) for port in ports) if isinstance(ports, list) else str(ports)])
    argv.extend(targets)
    remote = f"timeout {timeout} {shlex.join(argv)}"
    try:
        completed = _run_via_configured_pivot(remote_command=remote, timeout_seconds=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return _payload(
            success=False,
            stderr=str(exc),
            exit_code="pivoted_scan_failed",
            parsed={"runtime_hints": {"blocked_by": "pivoted_scan_failed"}},
        )
    if completed is None:
        return _payload(
            success=False,
            stderr="configured pivot transport unavailable",
            exit_code="pivot_unavailable",
            parsed={"runtime_hints": {"blocked_by": "pivot_unavailable"}},
        )
    if completed.returncode != 0 and "could not locate nse_main.lua" in str(completed.stderr):
        fallback_argv = [part for part in argv if part != "-sV"]
        fallback_remote = f"timeout {timeout} {shlex.join(fallback_argv)}"
        try:
            fallback = _run_via_configured_pivot(remote_command=fallback_remote, timeout_seconds=timeout)
        except (OSError, subprocess.SubprocessError):
            fallback = None
        if fallback is not None:
            completed = fallback
    stdout = _limit(completed.stdout, MAX_OUTPUT_CHARS)
    stderr = _limit(_redact_configured_secret(completed.stderr), MAX_OUTPUT_CHARS)
    parsed = _parse_nmap_output(",".join(targets), stdout)
    route_id = _string(arguments.get("route_id")) or (_configured_pivot_route_candidates()[0].get("route_id") if _configured_pivot_route_candidates() else None)
    evidence_id = f"pivoted-nmap::{route_id or 'configured-pivot'}::{hashlib.sha256(','.join(targets).encode()).hexdigest()[:12]}"
    parsed["evidence"].append({"kind": "pivoted_service_discovery", "evidence_id": evidence_id, "route_id": route_id, "targets": targets})
    conditions = ["internal_service_discovered_after_authorized_route", "restricted_zone_service_discovered"]
    if any(_is_data_service(port=int(item.get("port") or 0), service_name=str(item.get("service") or item.get("service_name") or "")) for item in parsed.get("entities", []) if isinstance(item, dict)):
        conditions.append("restricted_data_service_discovered")
    parsed["runtime_hints"] = {
        **(parsed.get("runtime_hints") if isinstance(parsed.get("runtime_hints"), dict) else {}),
        "route_id": route_id,
        "pivoted_scan": True,
        "satisfied_conditions": [{"condition": name, "evidence_ids": [evidence_id]} for name in conditions],
    }
    parsed["writeback_hints"] = {"observation_category": "pivoted_service_discovery", "route_id": route_id}
    success = completed.returncode == 0 and bool(parsed.get("entities"))
    return _payload(success=success, stdout=stdout, stderr=stderr, exit_code=completed.returncode, parsed=parsed)


def _controlled_data_read_proof(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "host")
    port = _int(arguments.get("port"), 0)
    service = (_string(arguments.get("service")) or _classify_service_name(port=port, protocol="tcp")).lower()
    if service not in {"postgres", "postgresql"}:
        return _payload(
            success=False,
            stderr=f"unsupported controlled data service: {service}",
            exit_code="unsupported_data_service",
            parsed={"runtime_hints": {"blocked_by": "unsupported_data_service"}},
        )

    route_id = _string(arguments.get("route_id"))
    session_id = _string(arguments.get("session_id"))
    timeout = max(1, min(_int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS), MAX_COMMAND_TIMEOUT_SECONDS))
    database = _string(arguments.get("database")) or _string(os.getenv("AEGRA_LAB_DB_NAME") or os.getenv("AEGRA_LAB_DB_DATABASE"))
    username = _string(arguments.get("username")) or _string(os.getenv("AEGRA_LAB_DB_USER"))
    password = _string(os.getenv("AEGRA_LAB_DB_PASSWORD"))
    query = _string(arguments.get("query")) or _string(os.getenv("AEGRA_LAB_DB_PROOF_QUERY")) or "select current_database(), current_user, version()"
    missing = [name for name, value in {"database": database, "username": username, "password": password}.items() if not value]
    if missing:
        return _payload(
            success=False,
            stderr=f"missing configured data-service parameter(s): {', '.join(missing)}",
            exit_code="missing_data_service_config",
            parsed={"runtime_hints": {"blocked_by": "missing_data_service_config", "missing": missing}},
        )

    if route_id or session_id or _pivot_transport_available():
        result = _run_psql_query_via_configured_pivot(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            query=query,
            timeout_seconds=timeout,
        )
    else:
        result = _run_local_psql_query(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            query=query,
            timeout_seconds=timeout,
        )

    parsed = _default_parsed()
    evidence_id = f"controlled-data-proof::{route_id or session_id or 'direct'}::{host}:{port}"
    if not result["success"]:
        parsed["runtime_hints"] = {
            "blocked_by": "data_read_failed",
            "route_id": route_id,
            "session_id": session_id,
        }
        return _payload(success=False, stderr=result["stderr"], exit_code="data_read_failed", parsed=parsed)

    rows = [line for line in result["stdout"].splitlines() if line.strip()]
    proof_material = "\n".join(rows)
    proof_hash = hashlib.sha256(proof_material.encode("utf-8", errors="replace")).hexdigest()
    entity = {
        "type": "ControlledDataReadProof",
        "service": service,
        "host": host,
        "port": port,
        "database": database,
        "row_count": len(rows),
        "proof_sha256": proof_hash,
        "route_id": route_id,
        "session_id": session_id,
    }
    parsed["entities"].append(entity)
    parsed["findings"].append({"kind": "ControlledDataReadProof", "summary": "controlled data service proof read", "evidence_refs": [evidence_id]})
    parsed["evidence"].append(
        {
            "kind": "controlled_data_read_proof",
            "evidence_id": evidence_id,
            "host": host,
            "port": port,
            "service": service,
            "row_count": len(rows),
            "proof_sha256": proof_hash,
        }
    )
    parsed["runtime_hints"] = {
        "goal_satisfied": True,
        "controlled_data_read": True,
        "route_id": route_id,
        "session_id": session_id,
        "goal_evidence_refs": [evidence_id],
        "satisfied_conditions": [
            {"condition": "controlled_data_service_read", "evidence_ids": [evidence_id]},
            {"condition": "goal_check_recorded", "evidence_ids": [evidence_id]},
        ],
    }
    parsed["writeback_hints"] = {"observation_category": "controlled_data_read_proof", "route_id": route_id}
    return _payload(success=True, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), parsed=parsed)


def _goal_check(arguments: dict[str, Any]) -> dict[str, Any]:
    url = _required(arguments, "url")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    expected_status = arguments.get("expected_status")
    body_contains = arguments.get("body_contains")
    response = _open_url(url, method="GET", timeout=timeout)
    checks: list[dict[str, Any]] = []
    if expected_status is not None:
        checks.append(
            {
                "type": "status",
                "expected": int(expected_status),
                "actual": response["status"],
                "passed": int(response["status"]) == int(expected_status),
            }
        )
    if body_contains is not None:
        needle = str(body_contains)
        checks.append({"type": "body_contains", "expected": needle, "passed": needle in response["body_excerpt"]})
    hidden_marker_check = _hidden_marker_check(arguments, response)
    if hidden_marker_check is not None:
        checks.append(hidden_marker_check)
    passed = all(item["passed"] for item in checks) if checks else int(response["status"]) < 400
    parsed = _parsed_http(url=url, response=response)
    finding = {"kind": "GoalCheck", "url": url, "goal_satisfied": passed, "checks": checks, "evidence_refs": []}
    parsed["findings"].append(finding)
    parsed["evidence"].append({"kind": "goal verification evidence", "url": url, "goal_satisfied": passed})
    parsed["runtime_hints"] = {"goal_satisfied": passed, "goal_summary": "goal check passed" if passed else "goal check did not pass", "goal_evidence_refs": []}
    parsed["writeback_hints"] = {"observation_category": "goal_check", "url": url, "goal_satisfied": passed}
    return _payload(
        success=passed,
        stdout=json.dumps(finding, ensure_ascii=True, sort_keys=True),
        exit_code=0 if passed else "goal_not_satisfied",
        parsed=parsed,
    )


def _artifact_store(arguments: dict[str, Any]) -> dict[str, Any]:
    name = _safe_artifact_name(str(arguments.get("name") or "artifact"))
    content_type = str(arguments.get("content_type") or "text/plain")
    if isinstance(arguments.get("json_content"), dict):
        suffix = ".json"
        content = json.dumps(arguments["json_content"], ensure_ascii=True, indent=2, sort_keys=True)
        content_type = "application/json"
    else:
        suffix = ".txt"
        content = str(arguments.get("content", ""))
    if len(content) > MAX_OUTPUT_CHARS:
        return _payload(success=False, stderr="artifact content exceeds max size", exit_code="artifact_too_large")
    artifact_dir = Path(os.getenv("AEGRA_MCP_ARTIFACT_DIR", "var/mcp_artifacts")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{name}{suffix}"
    path.write_text(content, encoding="utf-8")
    artifact = {"type": "file", "path": str(path), "content_type": content_type, "bytes": len(content.encode("utf-8"))}
    parsed = _default_parsed()
    parsed["writeback_hints"] = {"observation_category": "evidence_artifact", "artifact_path": str(path)}
    return _payload(success=True, stdout=json.dumps(artifact, ensure_ascii=True, sort_keys=True), artifacts=[artifact], parsed=parsed)


def _nuclei_scan(arguments: dict[str, Any]) -> dict[str, Any]:
    binary = _require_binary("nuclei")
    if binary is None:
        return _tool_unavailable("nuclei_scan", "nuclei")
    url = _required(arguments, "url")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    argv = [binary, "-u", url, "-jsonl", "-silent", "-no-color"]
    severities = _string_items(arguments.get("severity")) or ["low", "medium", "high", "critical"]
    argv.extend(["-severity", ",".join(severities)])
    rate_limit = _int(arguments.get("rate_limit"), 20)
    argv.extend(["-rl", str(rate_limit)])
    for template in _string_items(arguments.get("templates")):
        argv.extend(["-t", template])
    result = _run_command(_command_args_from_tool(arguments, argv=argv, timeout_seconds=timeout, max_output_chars=MAX_OUTPUT_CHARS))
    parsed = _parse_nuclei_jsonl(url, result.get("stdout", ""))
    result["parsed"] = parsed
    return result


def _whatweb_fingerprint(arguments: dict[str, Any]) -> dict[str, Any]:
    binary = _require_binary("whatweb")
    if binary is None:
        return _tool_unavailable("whatweb_fingerprint", "whatweb")
    url = _required(arguments, "url")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    result = _run_command(
        _command_args_from_tool(
            arguments,
            argv=[binary, "--no-errors", "--log-json=-", url],
            timeout_seconds=timeout,
        )
    )
    parsed = _parse_whatweb_json(url, result.get("stdout", ""))
    result["parsed"] = parsed
    return result


def _ffuf_discover(arguments: dict[str, Any]) -> dict[str, Any]:
    binary = _require_binary("ffuf")
    if binary is None:
        return _tool_unavailable("ffuf_discover", "ffuf")
    base_url = str(arguments.get("base_url") or arguments.get("url") or "").rstrip("/")
    if not base_url:
        raise ValueError("ffuf_discover requires base_url or url")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    wordlist = str(arguments.get("wordlist") or "").strip()
    temp_wordlist: Path | None = None
    if not wordlist:
        words = _string_items(arguments.get("words")) or DEFAULT_FFUF_WORDS
        temp_wordlist = _artifact_temp_path("ffuf-words", ".txt")
        temp_wordlist.write_text("\n".join(words) + "\n", encoding="utf-8")
        wordlist = str(temp_wordlist)
    match_status = ",".join(str(item) for item in (arguments.get("match_status") or [200, 204, 301, 302, 307, 401, 403]))
    argv = [binary, "-u", f"{base_url}/FUZZ", "-w", wordlist, "-mc", match_status, "-of", "json"]
    result = _run_command(_command_args_from_tool(arguments, argv=argv, timeout_seconds=timeout, max_output_chars=MAX_OUTPUT_CHARS))
    parsed = _parse_ffuf_json(base_url, result.get("stdout", ""))
    result["parsed"] = parsed
    if temp_wordlist is not None:
        try:
            temp_wordlist.unlink()
        except OSError:
            pass
    return result


_EXPLOIT_PROFILE_CACHE: dict[str, dict[str, Any]] = {}
_SAFE_OUTPUT_PREFIX = "/opt/aegra/"
_SAFE_OBSERVE_ZONES = {"flags", "hints", "loot"}


def _load_exploit_profile(profile_id: str) -> dict[str, Any] | None:
    """Load an exploit profile YAML from configs/exploit_profiles/. Cached per process."""
    if profile_id in _EXPLOIT_PROFILE_CACHE:
        return _EXPLOIT_PROFILE_CACHE[profile_id]
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "", profile_id)
    if not safe_id:
        return None
    candidate = Path("configs/exploit_profiles") / f"{safe_id}.yml"
    if not candidate.exists():
        candidate = Path("configs/exploit_profiles") / f"{safe_id}.yaml"
    if not candidate.exists():
        return None
    try:
        import yaml  # type: ignore[import-not-found]
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    _EXPLOIT_PROFILE_CACHE[safe_id] = payload
    return payload


def _lab_authorized_exploit_execute(arguments: dict[str, Any]) -> dict[str, Any]:
    profile_id = _required(arguments, "exploit_profile_id")
    target_url = _required(arguments, "target_url")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_COMMAND_TIMEOUT_SECONDS)

    profile = _load_exploit_profile(profile_id)
    if profile is None:
        return _payload(
            success=False,
            stderr=f"exploit profile not found: {profile_id}",
            exit_code="exploit_profile_not_found",
            parsed={"runtime_hints": {"blocked_by": "exploit_profile_not_found", "exploit_profile_id": profile_id}},
        )

    # Validate allowed_operations are present and non-destructive
    allowed_ops = profile.get("allowed_operations") or []
    if not allowed_ops:
        return _payload(
            success=False,
            stderr=f"exploit profile {profile_id} has no allowed_operations — execution blocked",
            exit_code="no_allowed_operations",
            parsed={"runtime_hints": {"blocked_by": "no_allowed_operations", "exploit_profile_id": profile_id}},
        )

    # Validate evidence_output_paths only use /opt/aegra/
    for output_path in profile.get("evidence_output_paths") or []:
        if not str(output_path).startswith(_SAFE_OUTPUT_PREFIX):
            return _payload(
                success=False,
                stderr=f"exploit profile {profile_id} references unsafe output path: {output_path}",
                exit_code="unsafe_output_path",
                parsed={"runtime_hints": {"blocked_by": "unsafe_output_path"}},
            )

    # Build substitution context for safe_payload_template
    session_id = _string(arguments.get("session_id"))
    route_id = _string(arguments.get("route_id"))
    extra_params = arguments.get("extra_params") or {}
    sub_context = {
        "target_url": target_url,
        "session_id": session_id or "",
        "route_id": route_id or "",
        **{str(k): str(v) for k, v in extra_params.items()},
    }

    # Execute safe_payload_template as a profile-declared bounded HTTP action.
    template = _string(profile.get("safe_payload_template"))
    result_stdout = ""
    result_success = False
    executed_steps: list[str] = []

    if template:
        safe_command = _string(extra_params.get("safe_command") or extra_params.get("safe_cmd"))
        if "SAFE_CMD" in template and safe_command is None:
            return _payload(
                success=False,
                stderr="safe_payload_template requires extra_params.safe_command",
                exit_code="safe_command_required",
                parsed={
                    "runtime_hints": {
                        "blocked_by": "safe_command_required",
                        "exploit_profile_id": profile_id,
                    }
                },
            )
        try:
            rendered_payload = template.format(**sub_context)
        except (KeyError, ValueError):
            rendered_payload = template
        if safe_command is not None:
            rendered_payload = rendered_payload.replace("SAFE_CMD", safe_command.replace("\\", "\\\\").replace("'", "\\'"))
        rendered_payload = " ".join(rendered_payload.split())
        if rendered_payload.lower().startswith("content-type:"):
            content_type = rendered_payload.split(":", 1)[1].strip()
            try:
                response = _open_url(
                    target_url,
                    method="POST",
                    timeout=timeout,
                    headers={"Content-Type": content_type},
                    data=b"",
                )
                body_excerpt = response["body_excerpt"]
                result_success = int(response["status"]) < 500
                command_marker = _string(extra_params.get("success_contains"))
                if command_marker:
                    result_success = result_success and command_marker in body_excerpt
                result_stdout = json.dumps(
                    {
                        "url": target_url,
                        "status": response["status"],
                        "body_excerpt": body_excerpt[:2048],
                        "body_excerpt_sha256": hashlib.sha256(
                            body_excerpt.encode("utf-8", errors="replace")
                        ).hexdigest(),
                    },
                    sort_keys=True,
                )
                executed_steps.append("http_content_type_payload")
            except Exception as exc:
                result_stdout = ""
                result_success = False
                executed_steps.append(f"http_content_type_payload_failed:{exc!s:.120}")
        else:
            try:
                response = _open_url(rendered_payload, method="GET", timeout=timeout)
                result_success = int(response["status"]) < 500
                result_stdout = json.dumps({
                    "url": rendered_payload,
                    "status": response["status"],
                    "body_excerpt_sha256": hashlib.sha256(
                        response["body_excerpt"].encode("utf-8", errors="replace")
                    ).hexdigest(),
                }, sort_keys=True)
                executed_steps.append(f"http_probe:{rendered_payload}")
            except Exception as exc:
                result_stdout = ""
                result_success = False
                executed_steps.append(f"http_probe_failed:{exc!s:.120}")
    else:
        # No payload template — probe the target URL directly as a bounded availability check
        try:
            response = _open_url(target_url, method="GET", timeout=timeout)
            result_success = int(response["status"]) < 500
            result_stdout = json.dumps({"url": target_url, "status": response["status"]}, sort_keys=True)
            executed_steps.append(f"availability_check:{target_url}")
        except Exception as exc:
            result_success = False
            executed_steps.append(f"availability_check_failed:{exc!s:.120}")

    capability_id = f"exploit-capability::{profile_id}::{_safe_artifact_name(target_url)}"
    parsed = _default_parsed()
    parsed["entities"].append({
        "type": "ExploitCapability",
        "capability_id": capability_id,
        "exploit_profile_id": profile_id,
        "target_url": target_url,
        "session_id": session_id,
        "route_id": route_id,
        "allowed_operations": allowed_ops,
        "executed_steps": executed_steps,
        "confidence": 0.8 if result_success else 0.3,
    })
    parsed["evidence"].append({
        "kind": "exploit_execution_evidence",
        "capability_id": capability_id,
        "exploit_profile_id": profile_id,
        "target_url": target_url,
        "success": result_success,
        "executed_steps": executed_steps,
    })
    parsed["runtime_hints"] = {
        "exploit_executed": result_success,
        "exploit_profile_id": profile_id,
        "capability_id": capability_id,
        "session_id": session_id,
        "route_id": route_id,
    }
    satisfied_conditions = extra_params.get("satisfied_conditions")
    if result_success and isinstance(satisfied_conditions, list):
        parsed["runtime_hints"]["satisfied_conditions"] = [str(item) for item in satisfied_conditions if str(item).strip()]
    parsed["writeback_hints"] = {
        "observation_category": "exploit_execution",
        "exploit_profile_id": profile_id,
        "target_url": target_url,
    }
    return _payload(success=result_success, stdout=result_stdout, exit_code=0 if result_success else "exploit_failed", parsed=parsed)


def _post_access_observe(arguments: dict[str, Any]) -> dict[str, Any]:
    zone = (_string(arguments.get("zone")) or "hints").lower()
    if zone not in _SAFE_OBSERVE_ZONES:
        return _payload(
            success=False,
            stderr=f"zone must be one of {sorted(_SAFE_OBSERVE_ZONES)}, got: {zone}",
            exit_code="invalid_zone",
            parsed={"runtime_hints": {"blocked_by": "invalid_zone"}},
        )

    base_path = Path(_SAFE_OUTPUT_PREFIX) / zone
    parsed = _default_parsed()
    artifacts_found: list[dict[str, Any]] = []

    if not base_path.exists():
        parsed["runtime_hints"] = {"zone": zone, "files_found": 0, "zone_exists": False}
        return _payload(success=True, stdout=json.dumps({"zone": zone, "files": []}, sort_keys=True), parsed=parsed)

    for entry in sorted(base_path.iterdir()):
        if not entry.is_file():
            continue
        try:
            raw = entry.read_bytes()
        except OSError:
            continue
        # For flags zone, only return SHA-256 hash (never the raw content)
        content_hash = hashlib.sha256(raw).hexdigest()
        item: dict[str, Any] = {
            "type": "PostAccessObservation",
            "zone": zone,
            "filename": entry.name,
            "size_bytes": len(raw),
            "content_sha256": content_hash,
        }
        if zone != "flags":
            # Hints and loot: return decoded text if it looks safe (no newlines → single-value marker, redact)
            text = raw.decode("utf-8", errors="replace").strip()
            # Redact anything that looks like a raw flag/token (short single-line with no spaces)
            if "\n" not in text and len(text) <= 64 and " " not in text:
                item["content_type"] = "redacted_marker"
                item["content_hash"] = content_hash
            else:
                item["content"] = text[:512]
                item["content_type"] = "text"
        artifacts_found.append(item)
        parsed["entities"].append(item)
        parsed["evidence"].append({"kind": "post_access_observation", "zone": zone, "filename": entry.name, "content_sha256": content_hash})

    parsed["runtime_hints"] = {"zone": zone, "files_found": len(artifacts_found), "zone_exists": True}
    parsed["writeback_hints"] = {"observation_category": "post_access_observation", "zone": zone}
    return _payload(success=True, stdout=json.dumps({"zone": zone, "files": artifacts_found}, ensure_ascii=True, sort_keys=True), parsed=parsed)


def _read_lab_marker(arguments: dict[str, Any]) -> dict[str, Any]:
    marker_id = _required(arguments, "marker_id")
    goal_id = _string(arguments.get("goal_id")) or marker_id

    # First try hidden fixture (already implemented)
    marker_literal = _hidden_marker_literal(marker_id)

    # Fall back to /opt/aegra/flags/{marker_id}
    if marker_literal is None:
        candidate = Path(_SAFE_OUTPUT_PREFIX) / "flags" / _safe_artifact_name(marker_id)
        if candidate.exists():
            try:
                marker_literal = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                marker_literal = None

    parsed = _default_parsed()
    if marker_literal is None:
        parsed["runtime_hints"] = {"marker_found": False, "marker_id": marker_id}
        return _payload(
            success=False,
            stderr=f"lab marker not found: {marker_id}",
            exit_code="marker_not_found",
            parsed=parsed,
        )

    # Compute HMAC submission hash — never return raw marker
    signing_key = os.environ.get("AEGRA_ORACLE_SIGNING_KEY", "aegra-default-dev-key")
    import hmac as _hmac
    payload_str = json.dumps({"goal_id": goal_id, "marker_id": marker_id}, sort_keys=True)
    submission_hash = _hmac.new(
        signing_key.encode("utf-8"),
        payload_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    proof_token = f"proof:{goal_id}:{submission_hash[:16]}"

    evidence_id = f"marker-evidence::{marker_id}"
    parsed["entities"].append({
        "type": "GoalCheck",
        "goal_id": goal_id,
        "marker_id": marker_id,
        "marker_found": True,
    })
    parsed["evidence"].append({
        "kind": "goal verification evidence",
        "evidence_id": evidence_id,
        "marker_id": marker_id,
        "goal_id": goal_id,
        "proof_token": proof_token,
    })
    parsed["runtime_hints"] = {
        "goal_satisfied": True,
        "marker_found": True,
        "marker_id": marker_id,
        "goal_id": goal_id,
        "proof_token": proof_token,
        "evidence_refs": [evidence_id],
    }
    parsed["writeback_hints"] = {"observation_category": "goal_marker_check", "marker_id": marker_id}
    result = {"marker_found": True, "marker_id": marker_id, "proof_token": proof_token, "evidence_id": evidence_id}
    return _payload(success=True, stdout=json.dumps(result, sort_keys=True), parsed=parsed)


def _pivot_route_register(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "destination_host")
    port = _int(arguments.get("destination_port"), 0)
    if port <= 0 or port > 65535:
        raise ValueError("destination_port must be between 1 and 65535")
    protocol = (_string(arguments.get("protocol")) or "tcp").lower()
    route_id = _string(arguments.get("route_id")) or (
        f"route::{_string(arguments.get('source_host')) or 'unknown'}::{host}:{port}"
    )
    zone_ref = _string(arguments.get("zone_ref"))
    parsed = _default_parsed()
    entity = {
        "type": "PivotRoute",
        "route_id": route_id,
        "source_host": _string(arguments.get("source_host")),
        "via_host": _string(arguments.get("via_host")),
        "destination_host": host,
        "port": port,
        "protocol": protocol,
        "zone_ref": zone_ref,
        "status": "registered",
        "confidence": 0.9,
    }
    parsed["entities"].append(entity)
    parsed["evidence"].append({"kind": "pivot_route_registration", "route_id": route_id, "destination_host": host, "port": port})
    parsed["runtime_hints"] = {
        "register_pivot_route": True,
        "route_id": route_id,
        "destination_host": host,
        "source_host": _string(arguments.get("source_host")),
        "via_host": _string(arguments.get("via_host")),
        "session_id": _string(arguments.get("session_id")),
        "allowed_ports": [port],
        "protocol": protocol,
        "zone_ref": zone_ref,
        "reachable": True,
    }
    parsed["writeback_hints"] = {"observation_category": "pivot_route_registration", "route_id": route_id}
    return _payload(success=True, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), parsed=parsed)


def _internal_goal_check(arguments: dict[str, Any]) -> dict[str, Any]:
    route_id = _string(arguments.get("route_id"))
    session_id = _string(arguments.get("session_id"))
    if route_id is None and session_id is None:
        return _payload(
            success=False,
            stderr="internal_goal_check requires route_id or session_id",
            exit_code="missing_route",
            parsed={"runtime_hints": {"blocked_by": "missing_route_or_session"}},
        )

    host = _required(arguments, "host")
    port = _int(arguments.get("port"), 80)
    protocol = (_string(arguments.get("protocol")) or "http").lower()
    path = str(arguments.get("path") or "/")
    if not path.startswith("/"):
        path = "/" + path
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    url = f"{protocol}://{host}:{port}{path}"

    try:
        response = _open_url(url, method="GET", timeout=timeout)
    except Exception as exc:
        parsed = _default_parsed()
        parsed["runtime_hints"] = {
            "goal_satisfied": False,
            "goal_summary": f"internal probe failed: {exc!s:.120}",
            "route_id": route_id,
            "session_id": session_id,
        }
        return _payload(success=False, stderr=str(exc), exit_code="probe_failed", parsed=parsed)

    checks: list[dict[str, Any]] = []
    expected_status = arguments.get("expected_status")
    body_contains = arguments.get("body_contains")
    if expected_status is not None:
        checks.append({
            "type": "status",
            "expected": int(expected_status),
            "actual": response["status"],
            "passed": int(response["status"]) == int(expected_status),
        })
    if body_contains is not None:
        needle = str(body_contains)
        checks.append({"type": "body_contains", "expected": needle, "passed": needle in response["body_excerpt"]})
    hidden_check = _hidden_marker_check(arguments, response)
    if hidden_check is not None:
        checks.append(hidden_check)
    passed = all(item["passed"] for item in checks) if checks else int(response["status"]) < 400

    evidence_id = f"internal-goal-evidence::{route_id or session_id}::{host}:{port}"
    parsed = _parsed_http(url=url, response=response)
    finding = {
        "kind": "GoalCheck",
        "url": url,
        "host": host,
        "port": port,
        "route_id": route_id,
        "goal_satisfied": passed,
        "checks": checks,
        "evidence_refs": [evidence_id],
    }
    parsed["findings"].append(finding)
    parsed["evidence"].append({
        "kind": "goal verification evidence",
        "evidence_id": evidence_id,
        "url": url,
        "goal_satisfied": passed,
        "route_id": route_id,
        "session_id": session_id,
    })
    parsed["runtime_hints"] = {
        "goal_satisfied": passed,
        "goal_summary": "internal goal check passed" if passed else "internal goal check did not pass",
        "goal_evidence_refs": [evidence_id],
        "route_id": route_id,
        "session_id": session_id,
    }
    parsed["writeback_hints"] = {"observation_category": "internal_goal_check", "url": url, "goal_satisfied": passed}
    return _payload(
        success=passed,
        stdout=json.dumps(finding, ensure_ascii=True, sort_keys=True),
        exit_code=0 if passed else "goal_not_satisfied",
        parsed=parsed,
    )


def _pivot_transport_available() -> bool:
    return bool(os.getenv("AEGRA_LAB_PIVOT_HOST") and os.getenv("AEGRA_LAB_PIVOT_USER") and os.getenv("AEGRA_LAB_PIVOT_PASSWORD"))


def _runtime_policy_payload() -> dict[str, Any]:
    path = _string(os.getenv("AEGRA_RUNTIME_POLICY_PATH"))
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _configured_pivot_route_candidates() -> list[dict[str, Any]]:
    policy = _runtime_policy_payload()
    adapter_policy = policy.get("adapter_policy") if isinstance(policy.get("adapter_policy"), dict) else {}
    pivot_policy = adapter_policy.get("pivot") if isinstance(adapter_policy.get("pivot"), dict) else {}
    candidates: list[dict[str, Any]] = []
    for value in pivot_policy.values():
        if not isinstance(value, dict):
            continue
        route_id = _string(value.get("route_id"))
        destination_cidr = _string(value.get("destination_cidr"))
        destination_host = _string(value.get("destination_host"))
        if route_id is None and destination_cidr is None and destination_host is None:
            continue
        transport = value.get("transport") if isinstance(value.get("transport"), dict) else {}
        candidates.append(
            {
                "route_id": route_id,
                "source_host": _string(value.get("source_host")),
                "via_host": _string(value.get("via_host")),
                "destination_cidr": destination_cidr,
                "destination_host": destination_host,
                "protocol": _string(value.get("protocol")),
                "transport_adapter": _string(transport.get("adapter")),
            }
        )
    return candidates


def _configured_pivot_route_by_id(route_id: str | None) -> dict[str, Any] | None:
    if not route_id:
        return None
    for candidate in _configured_pivot_route_candidates():
        if candidate.get("route_id") == route_id:
            return candidate
    return None


def _run_via_configured_pivot(*, remote_command: str, timeout_seconds: int) -> subprocess.CompletedProcess[str] | None:
    if not _pivot_transport_available() or shutil.which("ssh") is None or shutil.which("sshpass") is None:
        return None
    env = os.environ.copy()
    env["SSHPASS"] = str(os.getenv("AEGRA_LAB_PIVOT_PASSWORD") or "")
    argv = [
        "sshpass",
        "-e",
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={max(1, min(timeout_seconds, 30))}",
        f"{os.getenv('AEGRA_LAB_PIVOT_USER')}@{os.getenv('AEGRA_LAB_PIVOT_HOST')}",
        remote_command,
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds + 5, env=env)


def _probe_tcp_via_configured_pivot(*, host: str, port: int, timeout_seconds: int) -> dict[str, Any]:
    inner = f"nc -z -w {max(1, min(timeout_seconds, 30))} {shlex.quote(host)} {int(port)}"
    remote = f"timeout {max(1, min(timeout_seconds, 30))} sh -lc {shlex.quote(inner)}"
    try:
        completed = _run_via_configured_pivot(remote_command=remote, timeout_seconds=timeout_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"reachable": False, "stderr": str(exc)}
    if completed is None:
        return {"reachable": False, "stderr": "configured pivot transport unavailable"}
    return {"reachable": completed.returncode == 0, "stderr": _limit(completed.stderr, 500)}


def _run_psql_query_via_configured_pivot(
    *,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
    query: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = (
        f"PGPASSWORD={shlex.quote(password)} "
        f"psql -h {shlex.quote(host)} -p {int(port)} -U {shlex.quote(username)} "
        f"-d {shlex.quote(database)} -Atc {shlex.quote(query)}"
    )
    remote = f"timeout {max(1, min(timeout_seconds, 120))} sh -lc {shlex.quote(command)}"
    try:
        completed = _run_via_configured_pivot(remote_command=remote, timeout_seconds=timeout_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"success": False, "stdout": "", "stderr": str(exc)}
    if completed is None:
        return {"success": False, "stdout": "", "stderr": "configured pivot transport unavailable"}
    return {
        "success": completed.returncode == 0,
        "stdout": _limit(completed.stdout, 5000),
        "stderr": _limit(_redact_configured_secret(completed.stderr), 1000),
    }


def _run_local_psql_query(
    *,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
    query: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if shutil.which("psql") is None:
        return {"success": False, "stdout": "", "stderr": "psql is not installed"}
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    argv = ["psql", "-h", host, "-p", str(port), "-U", username, "-d", database, "-Atc", query]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"success": False, "stdout": "", "stderr": str(exc)}
    return {
        "success": completed.returncode == 0,
        "stdout": _limit(completed.stdout, 5000),
        "stderr": _limit(_redact_configured_secret(completed.stderr), 1000),
    }


def _redact_configured_secret(value: str) -> str:
    text = str(value or "")
    for secret in (os.getenv("AEGRA_LAB_PIVOT_PASSWORD"), os.getenv("AEGRA_LAB_DB_PASSWORD")):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _classify_service_name(*, port: int, protocol: str, hinted_service: str | None = None) -> str:
    if hinted_service:
        return hinted_service.lower()
    if port == 5432:
        return "postgres"
    if port == 3306:
        return "mysql"
    if port == 6379:
        return "redis"
    if protocol in {"http", "https"}:
        return protocol
    return "tcp"


def _is_data_service(*, port: int, service_name: str) -> bool:
    return service_name.lower() in {"postgres", "postgresql", "mysql", "mariadb", "redis", "mongodb", "mssql", "oracle"} or port in {
        5432,
        3306,
        6379,
        27017,
        1433,
        1521,
    }


def _success_condition_check(arguments: dict[str, Any]) -> dict[str, Any]:
    operation_id = _string(arguments.get("operation_id") or os.getenv("AEGRA_OPERATION_ID"))
    progress: dict[str, Any] = {}

    if operation_id:
        metadata_path = Path("var/runtime") / operation_id / "metadata.json"
        if metadata_path.exists():
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                progress = raw.get("success_condition_progress") or {}
            except (OSError, json.JSONDecodeError):
                progress = {}

    # Return only the redacted fields — never private_success_rubric, raw tokens, or proof values
    redacted: dict[str, Any] = {
        "eligible_for_stop": bool(progress.get("eligible_for_stop", False)),
        "satisfied": list(progress.get("satisfied") or []),
        "missing": list(progress.get("missing") or []),
        "chain_integrity": progress.get("chain_integrity"),
        "goal_proof_valid": progress.get("goal_proof_valid"),
    }
    parsed = _default_parsed()
    parsed["runtime_hints"] = {
        "success_condition_progress": redacted,
        "eligible_for_stop": redacted["eligible_for_stop"],
    }
    parsed["writeback_hints"] = {"observation_category": "success_condition_check"}
    return _payload(success=True, stdout=json.dumps(redacted, sort_keys=True), parsed=parsed)


def _parse_nmap_output(target: str, stdout: str) -> dict[str, Any]:
    parsed = _default_parsed()
    parsed["writeback_hints"] = {"observation_category": "service_discovery", "target": target}
    current_host = target
    host_ids: set[str] = set()

    def ensure_host(address: str) -> str:
        host_id = f"host::{address}"
        if host_id not in host_ids:
            parsed["entities"].append({"type": "Host", "id": host_id, "address": address, "confidence": 0.9})
            host_ids.add(host_id)
        return host_id

    ensure_host(target)
    for line in stdout.splitlines():
        host_match = re.search(r"^Nmap scan report for\s+(?P<host>.+?)\s*$", line)
        if host_match:
            current_host = _normalize_nmap_host(host_match.group("host")) or target
            ensure_host(current_host)
            continue
        match = re.search(r"(?P<port>\d+)/(?P<protocol>tcp|udp)\s+open\s+(?P<service>\S+)(?:\s+(?P<banner>.*))?", line)
        if not match:
            continue
        banner = (match.group("banner") or "").strip()
        product = banner.split()[0] if banner else ""
        version = banner.split()[1] if len(banner.split()) > 1 else ""
        host_id = ensure_host(current_host)
        service_id = f"service::{current_host}:{match.group('port')}/{match.group('protocol')}"
        entity = {
            "type": "Service",
            "id": service_id,
            "host": current_host,
            "port": int(match.group("port")),
            "protocol": match.group("protocol"),
            "service": match.group("service"),
            "product": product,
            "version": version,
            "banner": banner,
            "state": "open",
            "confidence": 0.85,
        }
        parsed["entities"].append(entity)
        parsed["relations"].append({"type": "HOSTS_SERVICE", "source_ref": {"graph": "kg", "ref_id": host_id, "ref_type": "Host"}, "target_ref": {"graph": "kg", "ref_id": service_id, "ref_type": "Service"}})
        parsed["evidence"].append({"kind": "scan evidence", "summary": line, "target": current_host, "confidence": 0.85})
    return parsed


def _normalize_nmap_host(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    paren_match = re.search(r"\((?P<address>(?:\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F:]+)\)", text)
    if paren_match:
        return paren_match.group("address")
    return text.split()[0] if text.split() else None


def _profile(profile_id: str) -> VulnerabilityProfile:
    try:
        return SAFE_VALIDATION_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown validation profile: {profile_id}") from exc


def _run_profile_prechecks(*, target_url: str, profile: VulnerabilityProfile, timeout: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    base_url = target_url.rstrip("/") + "/"
    for raw_path in profile.required_paths or ["/"]:
        path = "/" + str(raw_path).lstrip("/")
        url = urljoin(base_url, path.lstrip("/"))
        try:
            response = _open_url(url, method="GET", timeout=timeout)
            checks.append(
                {
                    "type": "http_path",
                    "url": url,
                    "status": response["status"],
                    "passed": int(response["status"]) < 500,
                }
            )
        except Exception as exc:
            checks.append({"type": "http_path", "url": url, "passed": False, "error": str(exc)})
    return checks


def _validation_payload(*, validation: ValidationResult, target_url: str, success: bool) -> dict[str, Any]:
    validation_payload = validation.model_dump(mode="json")
    parsed = _default_parsed()
    finding = {
        "kind": "ValidatedVulnerability" if validation.status == "validated" else "RejectedVulnerabilityCandidate",
        "vulnerability_id": validation.vulnerability_id,
        "candidate_id": validation.vulnerability_id,
        "status": validation.status,
        "confidence": validation.confidence,
        "evidence_refs": [],
    }
    parsed["findings"].append(finding)
    parsed["evidence"].append(
        {
            "kind": "validation evidence",
            "candidate_id": validation.vulnerability_id,
            "summary": validation.safe_payload_summary,
            "confidence": validation.confidence,
        }
    )
    parsed["runtime_hints"] = {
        "validated": validation.status == "validated",
        "requires_auth": False,
        "validation_status": validation.status,
        "vulnerability_id": validation.vulnerability_id,
    }
    parsed["writeback_hints"] = {"observation_category": "vulnerability_validation", "url": target_url}
    parsed["validation"] = validation_payload
    return _payload(
        success=success,
        stdout=json.dumps(finding, ensure_ascii=True, sort_keys=True),
        stderr=validation.failure_reason or "",
        exit_code=0 if success else validation.status,
        parsed=parsed,
    )


def _parsed_http(*, url: str, response: dict[str, Any]) -> dict[str, Any]:
    parsed = _default_parsed()
    body_excerpt = response.get("body_excerpt") or ""
    body_hash = hashlib.sha256(body_excerpt.encode("utf-8", errors="replace")).hexdigest()
    parsed["entities"].append(
        {
            "type": "WebEndpoint",
            "url": url,
            "status_code": response["status"],
            "title": _extract_title(body_excerpt),
            "server_header": response["headers"].get("server"),
            "content_type": response["headers"].get("content-type"),
            "body_excerpt_sha256": body_hash,
            "confidence": 0.8,
        }
    )
    parsed["evidence"].append({"kind": "http_probe", "url": url, "status_code": response["status"], "body_excerpt_sha256": body_hash})
    parsed["writeback_hints"] = {"observation_category": "http_probe", "url": url}
    return parsed


def _open_url(
    url: str,
    *,
    method: str,
    timeout: int,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> dict[str, Any]:
    request = Request(url, method=method, headers=headers or {}, data=data)
    try:
        # Lab tooling should reach local/container networks directly, regardless
        # of proxy variables inherited from the host shell.
        opener = build_opener(ProxyHandler({})).open
        with opener(request, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            headers = {key.lower(): value for key, value in response.headers.items()}
    except HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        status = int(exc.code)
        headers = {key.lower(): value for key, value in exc.headers.items()}
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    return {"status": status, "headers": headers, "body_excerpt": body[:4096]}


def _tool_unavailable(tool_name: str, binary: str) -> dict[str, Any]:
    return _payload(
        success=False,
        stderr=f"{binary} is not installed or not on PATH",
        exit_code="tool_unavailable",
        parsed={"runtime_hints": {"blocked_by": "tool_unavailable", "tool": tool_name, "missing_binary": binary}},
    )


def _require_binary(name: str) -> str | None:
    return shutil.which(name)


def _parse_nuclei_jsonl(url: str, stdout: str) -> dict[str, Any]:
    parsed = _default_parsed()
    parsed["writeback_hints"] = {"observation_category": "vulnerability_validation", "url": url, "tool": "nuclei"}
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        info = item.get("info") if isinstance(item.get("info"), dict) else {}
        finding = {
            "kind": "nuclei_finding",
            "template_id": item.get("template-id") or item.get("template_id"),
            "name": info.get("name"),
            "severity": info.get("severity"),
            "matched_at": item.get("matched-at") or item.get("matched_at"),
            "host": item.get("host"),
        }
        parsed["findings"].append(finding)
    return parsed


def _parse_whatweb_json(url: str, stdout: str) -> dict[str, Any]:
    parsed = _default_parsed()
    parsed["writeback_hints"] = {"observation_category": "web_fingerprint", "url": url, "tool": "whatweb"}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = []
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        plugins = item.get("plugins") if isinstance(item.get("plugins"), dict) else {}
        parsed["entities"].append({"type": "web_fingerprint", "url": item.get("target") or url, "plugins": sorted(plugins)})
    return parsed


def _parse_ffuf_json(base_url: str, stdout: str) -> dict[str, Any]:
    parsed = _default_parsed()
    parsed["writeback_hints"] = {"observation_category": "web_discovery", "base_url": base_url, "tool": "ffuf"}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = {}
    results = payload.get("results") if isinstance(payload, dict) and isinstance(payload.get("results"), list) else []
    for item in results:
        if not isinstance(item, dict):
            continue
        entity = {
            "type": "web_path",
            "url": item.get("url"),
            "path": "/" + str(item.get("input", {}).get("FUZZ", "")).lstrip("/") if isinstance(item.get("input"), dict) else None,
            "status": item.get("status"),
            "length": item.get("length"),
        }
        parsed["entities"].append(entity)
    return parsed


def _url_from_arguments(arguments: dict[str, Any]) -> str:
    url = arguments.get("url")
    if url:
        return str(url)
    target = _required(arguments, "target")
    scheme = str(arguments.get("scheme", "http"))
    port = arguments.get("port")
    path = str(arguments.get("path", "/"))
    if not path.startswith("/"):
        path = "/" + path
    netloc = f"{target}:{port}" if port else str(target)
    return f"{scheme}://{netloc}{path}"


def _payload(
    *,
    success: bool,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | str | None = None,
    parsed: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": 0 if exit_code is None and success else exit_code,
        "parsed": _merge_parsed(parsed),
        "artifacts": artifacts or [],
        "metadata": metadata or {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_raw_tool_output(*, arguments: dict[str, Any], payload: dict[str, Any]) -> str:
    operation_id = _string(arguments.get("operation_id") or os.getenv("AEGRA_OPERATION_ID"))
    trace_id = _string(arguments.get("trace_id")) or f"tool-{uuid.uuid4().hex}"
    if operation_id:
        output_dir = Path("var/runtime") / operation_id / "tool-outputs"
    else:
        output_dir = Path("var/runtime/_tool_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_artifact_name(trace_id)}.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)


def _ensure_raw_output_ref(*, arguments: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    existing_ref = _string(metadata.get("raw_output_ref") or payload.get("raw_output_ref"))
    if existing_ref:
        metadata["raw_output_ref"] = existing_ref
        payload["raw_output_ref"] = existing_ref
        parsed = payload.get("parsed")
        if isinstance(parsed, dict):
            hints = parsed.setdefault("writeback_hints", {})
            if isinstance(hints, dict):
                hints.setdefault("raw_output_ref", existing_ref)
        return payload
    if not _string(arguments.get("operation_id") or os.getenv("AEGRA_OPERATION_ID")):
        return payload
    raw_output_ref = _write_raw_tool_output(arguments=arguments, payload=payload)
    metadata["raw_output_ref"] = raw_output_ref
    payload["raw_output_ref"] = raw_output_ref
    parsed = payload.get("parsed")
    if isinstance(parsed, dict):
        hints = parsed.setdefault("writeback_hints", {})
        if isinstance(hints, dict):
            hints.setdefault("raw_output_ref", raw_output_ref)
    return payload


def _command_args_from_tool(
    arguments: dict[str, Any],
    *,
    argv: list[str],
    timeout_seconds: int,
    max_output_chars: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "argv": argv,
        "timeout_seconds": timeout_seconds,
        "operation_id": arguments.get("operation_id"),
        "trace_id": arguments.get("trace_id"),
    }
    if max_output_chars is not None:
        payload["max_output_chars"] = max_output_chars
    return payload


def _default_parsed() -> dict[str, Any]:
    return {
        "entities": [],
        "relations": [],
        "observations": [],
        "evidence": [],
        "findings": [],
        "runtime_hints": {},
        "writeback_hints": {},
    }


def _merge_parsed(parsed: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_parsed()
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            merged[key] = value
    return merged


def _lab_mode_enabled() -> bool:
    return os.getenv("AEGRA_LAB_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def load_hidden_fixture_from_env() -> dict[str, Any]:
    """Load the hidden MCP fixture without exposing it to Planner or Runtime.

    The fixture is consumed only inside MCP tool implementations. Tool responses
    may return booleans and generic evidence types derived from the fixture, but
    must never return credential secrets or marker literals.
    """

    global _HIDDEN_FIXTURE_CACHE
    fixture_path = _string(os.getenv("AEGRA_LAB_FIXTURE_PATH"))
    if fixture_path is None:
        return {}
    path = Path(fixture_path).expanduser().resolve()
    stat = path.stat()
    if (
        _HIDDEN_FIXTURE_CACHE is not None
        and _HIDDEN_FIXTURE_CACHE[0] == path
        and _HIDDEN_FIXTURE_CACHE[1] == stat.st_mtime
    ):
        return dict(_HIDDEN_FIXTURE_CACHE[2])
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load hidden YAML lab fixtures") from exc
        payload = yaml.safe_load(text)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("hidden lab fixture must be a mapping")
    _HIDDEN_FIXTURE_CACHE = (path, stat.st_mtime, dict(payload))
    return dict(payload)


def _hidden_marker_check(arguments: dict[str, Any], response: dict[str, Any]) -> dict[str, Any] | None:
    marker_id = _string(arguments.get("fixture_marker_id"))
    if marker_id is None:
        return None
    marker_literal = _hidden_marker_literal(marker_id)
    matched = bool(marker_literal and marker_literal in str(response.get("body_excerpt") or ""))
    return {
        "type": "hidden_marker",
        "fixture_marker_id": marker_id,
        "evidence_type": "goal_check",
        "matched": matched,
        "passed": matched,
        "generic_reason": "hidden marker matched" if matched else "hidden marker did not match",
    }


def _hidden_marker_literal(marker_id: str) -> str | None:
    fixture = load_hidden_fixture_from_env()
    markers = fixture.get("markers")
    if isinstance(markers, dict):
        value = markers.get(marker_id)
        if isinstance(value, dict):
            return _string(value.get("literal") or value.get("value") or value.get("marker"))
        return _string(value)
    if isinstance(markers, list):
        for item in markers:
            if not isinstance(item, dict):
                continue
            item_id = _string(item.get("id") or item.get("marker_id") or item.get("name"))
            if item_id == marker_id:
                return _string(item.get("literal") or item.get("value") or item.get("marker"))
    return None


def _required(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} is required")
    return str(value)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _limit(text: str, limit: int) -> str:
    return text[: max(0, limit)]


def _string_items(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_artifact_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return safe[:80] or "artifact"


def _artifact_temp_path(name: str, suffix: str) -> Path:
    artifact_dir = Path(os.getenv("AEGRA_MCP_ARTIFACT_DIR", "var/mcp_artifacts")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{_safe_artifact_name(name)}{suffix}"


def _extract_title(html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


__all__ = ["LAB_TOOL_SPECS", "call_lab_tool", "lab_tool_specs", "load_hidden_fixture_from_env"]
