"""Lab-only MCP tools used by the experimental LLM worker path."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import ssl
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_DISCOVERY_PATHS = ["/", "/robots.txt", "/sitemap.xml", "/admin", "/login"]
DEFAULT_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 20000
DEFAULT_FFUF_WORDS = ["admin", "login", "robots.txt", "sitemap.xml", "api", "debug", "health"]


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
        "name": "goal_check",
        "description": "Check a bounded goal condition such as HTTP status or body substring.",
        "inputSchema": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "expected_status": {"type": "integer"},
                "body_contains": {"type": "string"},
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
    timeout = _int(arguments.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    max_output = _int(arguments.get("max_output_chars"), MAX_OUTPUT_CHARS)
    cwd = arguments.get("cwd")
    env = os.environ.copy()
    if isinstance(arguments.get("env"), dict):
        env.update({str(key): str(value) for key, value in arguments["env"].items()})

    argv = arguments.get("argv")
    if isinstance(argv, list) and argv:
        completed = subprocess.run(
            [str(part) for part in argv],
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        command_label = " ".join(str(part) for part in argv)
    else:
        command = str(arguments.get("command", "")).strip()
        if not command:
            return _payload(success=False, stderr="run_command requires command or argv", exit_code="missing_command")
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            encoding="utf-8",
            errors="replace",
        )
        command_label = command

    parsed = _default_parsed()
    parsed["writeback_hints"] = {"observation_category": "command_execution", "command": command_label}
    return _payload(
        success=completed.returncode == 0,
        stdout=_limit(completed.stdout, max_output),
        stderr=_limit(completed.stderr, max_output),
        exit_code=completed.returncode,
        parsed=parsed,
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
    parsed["writeback_hints"] = {"observation_category": "credential_validation", "url": url, "username": username}
    return _payload(
        success=authenticated,
        stdout=json.dumps(finding, ensure_ascii=True, sort_keys=True),
        exit_code=response["status"],
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
    passed = all(item["passed"] for item in checks) if checks else int(response["status"]) < 400
    parsed = _parsed_http(url=url, response=response)
    finding = {"kind": "goal_condition", "url": url, "passed": passed, "checks": checks}
    parsed["findings"].append(finding)
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
    parsed["entities"].append({"type": "host", "address": target})
    for line in stdout.splitlines():
        match = re.search(r"(?P<port>\d+)/(?:tcp|udp)\s+open\s+(?P<service>\S+)(?:\s+(?P<banner>.*))?", line)
        if not match:
            continue
        entity = {
            "type": "service",
            "host": target,
            "port": int(match.group("port")),
            "service": match.group("service"),
            "banner": (match.group("banner") or "").strip(),
            "state": "open",
        }
        parsed["entities"].append(entity)
        parsed["relations"].append({"type": "HOSTS", "source": target, "target": f"{target}:{entity['port']}"})
    return parsed


def _parsed_http(*, url: str, response: dict[str, Any]) -> dict[str, Any]:
    parsed = _default_parsed()
    parsed["entities"].append(
        {
            "type": "http_endpoint",
            "url": url,
            "status": response["status"],
            "server": response["headers"].get("server"),
            "content_type": response["headers"].get("content-type"),
        }
    )
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
) -> dict[str, Any]:
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": 0 if exit_code is None and success else exit_code,
        "parsed": _merge_parsed(parsed),
        "artifacts": artifacts or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _default_parsed() -> dict[str, Any]:
    return {
        "entities": [],
        "relations": [],
        "findings": [],
        "writeback_hints": {},
        "runtime_hints": {},
    }


def _merge_parsed(parsed: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_parsed()
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            merged[key] = value
    return merged


def _lab_mode_enabled() -> bool:
    return os.getenv("AEGRA_LAB_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


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


__all__ = ["LAB_TOOL_SPECS", "call_lab_tool"]
