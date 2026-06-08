"""Lab-only MCP tools used by the experimental LLM worker path."""

from __future__ import annotations

import json
import os
import re
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
from urllib.request import Request, urlopen

from src.core.validation import ValidationPlan, ValidationResult, VulnerabilityProfile


DEFAULT_DISCOVERY_PATHS = ["/", "/robots.txt", "/sitemap.xml", "/admin", "/login"]
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_COMMAND_TIMEOUT_SECONDS = 120
MAX_COMMAND_TIMEOUT_SECONDS = 300
MAX_OUTPUT_CHARS = 20000
DEFAULT_FFUF_WORDS = ["admin", "login", "robots.txt", "sitemap.xml", "api", "debug", "health"]
SAFE_VALIDATION_PROFILES: dict[str, VulnerabilityProfile] = {
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
        "description": "Run an nmap service discovery scan against one lab target.",
        "inputSchema": {
            "type": "object",
            "required": ["target"],
            "properties": {
                "target": {"type": "string"},
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
        "description": "Probe one internal lab service using bounded TCP or HTTP checks.",
        "inputSchema": {
            "type": "object",
            "required": ["host", "port"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "protocol": {"type": "string", "default": "tcp"},
                "path": {"type": "string", "default": "/"},
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
        return _payload(
            success=False,
            stderr="AEGRA_LAB_MODE=1 is required before lab MCP tools can execute",
            exit_code="lab_mode_required",
            parsed={"runtime_hints": {"required_env": "AEGRA_LAB_MODE=1"}},
        )

    try:
        if name == "run_command":
            return _run_command(arguments)
        if name == "nmap_scan":
            return _nmap_scan(arguments)
        if name == "http_probe":
            return _http_probe(arguments)
        if name == "web_fingerprint":
            return _web_fingerprint(arguments)
        if name == "web_discover":
            return _web_discover(arguments)
        if name == "dns_lookup":
            return _dns_lookup(arguments)
        if name == "tls_probe":
            return _tls_probe(arguments)
        if name == "tcp_connect_probe":
            return _tcp_connect_probe(arguments)
        if name == "http_basic_auth_check":
            return _http_basic_auth_check(arguments)
        if name == "vuln_profile_match":
            return _vuln_profile_match(arguments)
        if name == "validation_precheck":
            return _validation_precheck(arguments)
        if name == "safe_vuln_validate":
            return _safe_vuln_validate(arguments)
        if name == "credential_check":
            return _credential_check(arguments)
        if name == "session_probe":
            return _session_probe(arguments)
        if name == "session_open_lab":
            return _session_open_lab(arguments)
        if name == "identity_context_probe":
            return _identity_context_probe(arguments)
        if name == "privilege_context_probe":
            return _privilege_context_probe(arguments)
        if name == "pivot_route_probe":
            return _pivot_route_probe(arguments)
        if name == "internal_service_discover":
            return _internal_service_discover(arguments)
        if name == "chain_goal_check":
            return _goal_check(arguments)
        if name == "goal_check":
            return _goal_check(arguments)
        if name == "artifact_store":
            return _artifact_store(arguments)
        if name == "nuclei_scan":
            return _nuclei_scan(arguments)
        if name == "whatweb_fingerprint":
            return _whatweb_fingerprint(arguments)
        if name == "ffuf_discover":
            return _ffuf_discover(arguments)
    except Exception as exc:
        return _payload(success=False, stderr=str(exc), exit_code="tool_error")
    return _payload(success=False, stderr=f"unknown lab MCP tool: {name}", exit_code="unknown_tool")


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
    target = _required(arguments, "target")
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
    argv.append(str(target))
    result = _run_command({"argv": argv, "timeout_seconds": timeout})
    parsed = _parse_nmap_output(str(target), result.get("stdout", ""))
    result["parsed"] = parsed
    return result


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
    parsed = _default_parsed()
    matches: list[dict[str, Any]] = []
    for profile in SAFE_VALIDATION_PROFILES.values():
        if service and profile.required_service and service.lower() != profile.required_service.lower():
            continue
        if product and profile.affected_products and not any(product in item.lower() for item in profile.affected_products):
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
    parsed = _default_parsed()
    entity = {"type": "identity_context", "host": host, "identity": identity, "session_id": _string(arguments.get("session_id"))}
    parsed["entities"].append(entity)
    parsed["runtime_hints"] = {"identity": identity, "host": host, "identity_context_observed": True}
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
            reachable = False
            error = str(exc)
    route_id = _string(arguments.get("route_id")) or f"route::{_string(arguments.get('source_host')) or 'unknown-source'}::{host}:{port}"
    parsed = _default_parsed()
    entity = {
        "type": "PivotRoute",
        "route_id": route_id,
        "source_host": _string(arguments.get("source_host")),
        "via_host": _string(arguments.get("via_host")),
        "destination_host": host,
        "port": port,
        "protocol": protocol,
        "status": "validated" if reachable else "unreachable",
        "confidence": 0.85 if reachable else 0.25,
    }
    parsed["entities"].append(entity)
    parsed["evidence"].append({"kind": "reachability evidence", "route_id": route_id, "destination_host": host, "port": port, "reachable": reachable})
    parsed["runtime_hints"] = {
        "register_pivot_route": True,
        "route_id": route_id,
        "destination_host": host,
        "source_host": _string(arguments.get("source_host")),
        "via_host": _string(arguments.get("via_host")),
        "session_id": _string(arguments.get("session_id")),
        "allowed_ports": [port],
        "protocol": protocol,
        "reachable": reachable,
        "reason": error,
    }
    parsed["writeback_hints"] = {"observation_category": "pivot_route_validation"}
    return _payload(success=reachable, stdout=json.dumps(entity, ensure_ascii=True, sort_keys=True), stderr=error, exit_code=0 if reachable else "unreachable", parsed=parsed)


def _internal_service_discover(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "host")
    port = _int(arguments.get("port"), 0)
    protocol = (_string(arguments.get("protocol")) or "tcp").lower()
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
        result = _tcp_connect_probe({"host": host, "port": port, "timeout_seconds": arguments.get("timeout_seconds")})
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else _default_parsed()
    parsed["entities"].append(
        {
            "type": "InternalService",
            "host": host,
            "port": port,
            "protocol": protocol,
            "via_pivot_route": _string(arguments.get("route_id")),
            "confidence": 0.75 if result.get("success") else 0.25,
        }
    )
    parsed["evidence"].append({"kind": "reachability evidence", "host": host, "port": port, "via_pivot_route": _string(arguments.get("route_id"))})
    result["parsed"] = parsed
    return result


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
    result = _run_command({"argv": argv, "timeout_seconds": timeout, "max_output_chars": MAX_OUTPUT_CHARS})
    parsed = _parse_nuclei_jsonl(url, result.get("stdout", ""))
    result["parsed"] = parsed
    return result


def _whatweb_fingerprint(arguments: dict[str, Any]) -> dict[str, Any]:
    binary = _require_binary("whatweb")
    if binary is None:
        return _tool_unavailable("whatweb_fingerprint", "whatweb")
    url = _required(arguments, "url")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    result = _run_command({"argv": [binary, "--no-errors", "--log-json=-", url], "timeout_seconds": timeout})
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
    result = _run_command({"argv": argv, "timeout_seconds": timeout, "max_output_chars": MAX_OUTPUT_CHARS})
    parsed = _parse_ffuf_json(base_url, result.get("stdout", ""))
    result["parsed"] = parsed
    if temp_wordlist is not None:
        try:
            temp_wordlist.unlink()
        except OSError:
            pass
    return result


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


def _open_url(url: str, *, method: str, timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(url, method=method, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
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
