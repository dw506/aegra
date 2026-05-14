"""Probe adapter layer for recon workers.

Adapters encapsulate command construction and output parsing for external
reconnaissance tools while keeping ``ReconWorker`` protocol-compatible.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.workers.tool_runner import ToolExecutionResult


class ProbeAdapterUnavailable(RuntimeError):
    """Raised when an adapter cannot build a usable command from the metadata."""


class ParsedProbeResult(BaseModel):
    """Normalized parsed output shared across probe adapters."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reachable: bool = False
    success: bool = False
    partial_success: bool = False
    blocked: bool = False
    blocked_reason: str | None = None
    failure_reason: str | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    runtime_hints: dict[str, Any] = Field(default_factory=dict)
    hosts: list[dict[str, Any]] = Field(default_factory=list)
    service: dict[str, Any] = Field(default_factory=dict)
    identity_context: dict[str, Any] = Field(default_factory=dict)
    raw_output: str | None = None


class ProbeAdapter(ABC):
    """Base class for command-building and parsing adapters."""

    adapter_name: str

    def can_handle(self, *, mode: str) -> bool:
        return True

    @abstractmethod
    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        """Return the subprocess command for this adapter."""

    def acceptable_exit_codes(self, *, mode: str, metadata: dict[str, Any]) -> set[int]:
        return {0}

    @abstractmethod
    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        """Convert raw tool output into the unified parsed probe structure."""

    def unavailable_result(self, *, target_hint: str, mode: str, reason: str) -> ParsedProbeResult:
        return ParsedProbeResult(
            summary=f"{self.adapter_name} unavailable for {mode} on {target_hint}",
            confidence=0.0,
            reachable=False,
            success=False,
            blocked=True,
            blocked_reason=reason,
            evidence={"adapter": self.adapter_name, "reason": reason},
            runtime_hints={"blocked_by": "tool_unavailable"},
        )


class CustomProbeAdapter(ProbeAdapter):
    """Adapter for arbitrary externally-provided probe commands that emit JSON."""

    adapter_name = "custom"

    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        raw = metadata.get("tool_command") or metadata.get("custom_probe_command")
        if not isinstance(raw, list) or not raw:
            raise ProbeAdapterUnavailable("tool_command or custom_probe_command list is required")
        return [str(item) for item in raw]

    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        trimmed = execution_result.stdout.strip()
        if execution_result.category == "command_not_found":
            return self.unavailable_result(target_hint=target_hint, mode=mode, reason=execution_result.error_message or "")
        if execution_result.category in {"timeout", "process_error"}:
            return ParsedProbeResult(
                summary=f"custom probe blocked for {target_hint}",
                confidence=0.0,
                reachable=False,
                success=False,
                blocked=True,
                blocked_reason=execution_result.category,
                failure_reason=execution_result.error_message,
                evidence={"adapter": self.adapter_name, "tool": execution_result.to_payload()},
                runtime_hints={"blocked_by": execution_result.category},
            )
        if not trimmed:
            return ParsedProbeResult(
                summary=f"custom probe returned no output for {target_hint}",
                confidence=0.0,
                reachable=False,
                success=False,
                blocked=False,
                failure_reason=execution_result.stderr.strip() or execution_result.error_message,
                evidence={"adapter": self.adapter_name, "stderr": execution_result.stderr, "tool": execution_result.to_payload()},
                runtime_hints={},
            )
        try:
            payload = json.loads(trimmed)
        except json.JSONDecodeError:
            return ParsedProbeResult(
                summary=trimmed,
                confidence=0.0 if not execution_result.success else 0.5,
                reachable=execution_result.success,
                success=execution_result.success,
                partial_success=bool(trimmed and not execution_result.success),
                blocked=False,
                failure_reason=(None if execution_result.success else execution_result.stderr.strip() or execution_result.error_message),
                evidence={"adapter": self.adapter_name, "stderr": execution_result.stderr, "tool": execution_result.to_payload()},
                runtime_hints={},
                raw_output=trimmed,
            )
        if not isinstance(payload, dict):
            payload = {"raw_output": payload}
        return _normalize_payload(
            payload=payload,
            target_hint=target_hint,
            mode=mode,
            adapter_name=self.adapter_name,
            execution_result=execution_result,
            default_confidence=float(metadata.get("confidence", 0.8)),
        )


class NmapAdapter(ProbeAdapter):
    """Adapter for text-mode nmap output."""

    adapter_name = "nmap"
    _report_re = re.compile(r"^Nmap scan report for (?P<label>.+)$", re.MULTILINE)
    _port_re = re.compile(
        r"^(?P<port>\d+)/(?:tcp|udp)\s+(?P<state>\S+)\s+(?P<service>\S+)(?:\s+(?P<banner>.+))?$",
        re.MULTILINE,
    )

    def can_handle(self, *, mode: str) -> bool:
        return mode in {"host_discovery", "host_discovery_result", "service_validation", "service_validation_result"}

    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        port = metadata.get("service_port") or metadata.get("port")
        command = [str(metadata.get("nmap_path", "nmap"))]
        extra_args = metadata.get("nmap_args")
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args)
        else:
            command.extend(["-n", "-Pn"])
            if mode in {"service_validation", "service_validation_result"}:
                command.append("-sV")
                if port is not None:
                    command.extend(["-p", str(port)])
        command.append(str(target_hint))
        return command

    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        if execution_result.category == "command_not_found":
            return self.unavailable_result(target_hint=target_hint, mode=mode, reason=execution_result.error_message or "")
        if execution_result.category in {"timeout", "process_error"}:
            return ParsedProbeResult(
                summary=f"nmap blocked for {target_hint}",
                confidence=0.0,
                reachable=False,
                success=False,
                blocked=True,
                blocked_reason=execution_result.category,
                failure_reason=execution_result.error_message,
                evidence={"adapter": self.adapter_name, "tool": execution_result.to_payload()},
                runtime_hints={"blocked_by": execution_result.category},
            )
        stdout = execution_result.stdout.strip()
        host_label = target_hint
        report = self._report_re.search(stdout)
        if report:
            host_label = report.group("label").strip()
        reachable = "Host is up" in stdout or "open" in stdout
        services: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        for match in self._port_re.finditer(stdout):
            port = int(match.group("port"))
            state = match.group("state").lower()
            protocol = "tcp" if f"{port}/tcp" in match.group(0) else "udp"
            service_name = match.group("service")
            banner = (match.group("banner") or "").strip() or None
            service_id = f"{host_label}:{port}/{protocol}"
            service = {
                "id": service_id,
                "type": "Service",
                "host_id": host_label,
                "port": port,
                "protocol": protocol,
                "service_name": service_name,
                "banner": banner,
                "state": state,
                "validated": state == "open",
            }
            services.append(service)
            relations.append(
                {
                    "type": "HOSTS",
                    "source": host_label,
                    "target": service_id,
                    "attributes": {"port": port, "protocol": protocol, "state": state},
                }
            )
        hosts = [{"host_id": host_label, "status": "up" if reachable else "unknown"}]
        primary_service = services[0] if services else {}
        payload = {
            "summary": (
                f"nmap service validation completed for {host_label}"
                if mode in {"service_validation", "service_validation_result"}
                else f"nmap host discovery completed for {host_label}"
            ),
            "confidence": float(metadata.get("confidence", 0.85 if services else 0.7)),
            "reachable": reachable,
            "success": execution_result.success and (reachable or bool(services)),
            "partial_success": (not execution_result.success) and bool(services),
            "failure_reason": (None if execution_result.success else execution_result.stderr.strip() or execution_result.error_message),
            "entities": [{"id": host_label, "type": "Host", "status": hosts[0]["status"]}, *services],
            "relations": relations,
            "evidence": {
                "adapter": self.adapter_name,
                "stdout_excerpt": stdout[:400],
                "stderr": execution_result.stderr,
                "tool": execution_result.to_payload(),
            },
            "runtime_hints": {
                "reachable": reachable,
                "discovered_open_ports": [service["port"] for service in services if service["validated"]],
            },
            "hosts": hosts,
            "service": primary_service,
        }
        return _normalize_payload(
            payload=payload,
            target_hint=target_hint,
            mode=mode,
            adapter_name=self.adapter_name,
            execution_result=execution_result,
            default_confidence=float(metadata.get("confidence", 0.8)),
        )


class MasscanAdapter(ProbeAdapter):
    """Adapter for masscan output."""

    adapter_name = "masscan"
    _open_re = re.compile(
        r"(?:Discovered open port|open)\s+(?P<port>\d+)/(?:tcp|udp)\s+(?:on\s+)?(?P<host>\S+)",
        re.IGNORECASE,
    )

    def can_handle(self, *, mode: str) -> bool:
        return mode in {"host_discovery", "host_discovery_result", "service_validation", "service_validation_result"}

    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        port = metadata.get("service_port") or metadata.get("port")
        command = [str(metadata.get("masscan_path", "masscan")), str(target_hint)]
        extra_args = metadata.get("masscan_args")
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args)
        else:
            command.extend(["--rate", str(metadata.get("masscan_rate", 1000))])
            if port is not None:
                command.extend(["-p", str(port)])
        return command

    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        if execution_result.category == "command_not_found":
            return self.unavailable_result(target_hint=target_hint, mode=mode, reason=execution_result.error_message or "")
        if execution_result.category in {"timeout", "process_error"}:
            return ParsedProbeResult(
                summary=f"masscan blocked for {target_hint}",
                confidence=0.0,
                reachable=False,
                success=False,
                blocked=True,
                blocked_reason=execution_result.category,
                failure_reason=execution_result.error_message,
                evidence={"adapter": self.adapter_name, "tool": execution_result.to_payload()},
                runtime_hints={"blocked_by": execution_result.category},
            )
        stdout = execution_result.stdout.strip()
        services: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        host_id = target_hint
        for match in self._open_re.finditer(stdout):
            port = int(match.group("port"))
            host_id = match.group("host")
            service_id = f"{host_id}:{port}/tcp"
            service = {
                "id": service_id,
                "type": "Service",
                "host_id": host_id,
                "port": port,
                "protocol": "tcp",
                "service_name": metadata.get("service_name"),
                "banner": None,
                "state": "open",
                "validated": True,
            }
            services.append(service)
            relations.append(
                {
                    "type": "HOSTS",
                    "source": host_id,
                    "target": service_id,
                    "attributes": {"port": port, "protocol": "tcp", "state": "open"},
                }
            )
        payload = {
            "summary": f"masscan completed for {host_id}",
            "confidence": float(metadata.get("confidence", 0.78 if services else 0.5)),
            "reachable": bool(services),
            "success": execution_result.success and bool(services),
            "partial_success": (not execution_result.success) and bool(services),
            "failure_reason": (None if execution_result.success else execution_result.stderr.strip() or execution_result.error_message),
            "entities": [{"id": host_id, "type": "Host", "status": "up" if services else "unknown"}, *services],
            "relations": relations,
            "evidence": {
                "adapter": self.adapter_name,
                "stdout_excerpt": stdout[:400],
                "stderr": execution_result.stderr,
                "tool": execution_result.to_payload(),
            },
            "runtime_hints": {
                "reachable": bool(services),
                "discovered_open_ports": [service["port"] for service in services],
            },
            "hosts": [{"host_id": host_id, "status": "up" if services else "unknown"}],
            "service": services[0] if services else {},
        }
        return _normalize_payload(
            payload=payload,
            target_hint=target_hint,
            mode=mode,
            adapter_name=self.adapter_name,
            execution_result=execution_result,
            default_confidence=float(metadata.get("confidence", 0.7)),
        )


class NucleiSafeTemplateAdapter(ProbeAdapter):
    """Adapter for nuclei JSONL output constrained to safe templates by default."""

    adapter_name = "nuclei"
    blocked_tags = {"destructive", "intrusive", "exploit"}

    def can_handle(self, *, mode: str) -> bool:
        return mode in {"vulnerability_validation", "vulnerability_validation_result", "service_validation", "service_validation_result"}

    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        blocked = self._blocked_template_tags(metadata)
        if blocked:
            raise ProbeAdapterUnavailable(f"nuclei template rejected by safe policy: {', '.join(sorted(blocked))}")
        command = [str(metadata.get("nuclei_path", "nuclei")), "-u", str(metadata.get("target_url", target_hint)), "-jsonl"]
        templates = metadata.get("nuclei_templates") or metadata.get("templates")
        if isinstance(templates, list):
            for template in templates:
                command.extend(["-t", str(template)])
        elif isinstance(templates, str) and templates.strip():
            command.extend(["-t", templates.strip()])
        tags = metadata.get("nuclei_include_tags") or metadata.get("safe_tags")
        if isinstance(tags, list) and tags:
            command.extend(["-tags", ",".join(str(tag) for tag in tags)])
        return command

    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        blocked = self._blocked_template_tags(metadata)
        if blocked:
            return ParsedProbeResult(
                summary=f"nuclei template rejected for {target_hint}",
                confidence=0.0,
                reachable=False,
                success=False,
                blocked=True,
                blocked_reason="unsafe_nuclei_template",
                evidence={"adapter": self.adapter_name, "blocked_tags": sorted(blocked), "tool": execution_result.to_payload()},
                runtime_hints={"blocked_by": "tool_policy"},
            )
        if execution_result.category == "command_not_found":
            return self.unavailable_result(target_hint=target_hint, mode=mode, reason=execution_result.error_message or "")
        findings: list[dict[str, Any]] = []
        for item in _json_lines(execution_result.stdout):
            info = item.get("info") if isinstance(item.get("info"), dict) else {}
            findings.append(
                {
                    "template_id": item.get("template-id") or item.get("template_id"),
                    "name": info.get("name") or item.get("matcher-name"),
                    "severity": info.get("severity"),
                    "matched_at": item.get("matched-at") or item.get("host") or target_hint,
                    "tags": sorted(_coerce_tags(info.get("tags"))),
                }
            )
        payload = {
            "summary": f"nuclei safe scan found {len(findings)} finding(s) for {target_hint}",
            "confidence": float(metadata.get("confidence", 0.8 if findings else 0.55)),
            "reachable": execution_result.success or bool(findings),
            "success": execution_result.success,
            "partial_success": bool(findings) and not execution_result.success,
            "failure_reason": None if execution_result.success else execution_result.stderr.strip() or execution_result.error_message,
            "entities": [
                {
                    "id": f"vuln::{finding.get('template_id') or index}::{target_hint}",
                    "type": "VulnerabilityCandidate",
                    **finding,
                }
                for index, finding in enumerate(findings, start=1)
            ],
            "evidence": {"adapter": self.adapter_name, "findings": findings, "tool": execution_result.to_payload()},
            "runtime_hints": {"nuclei_findings": len(findings), "safe_templates_only": True},
        }
        return _normalize_payload(
            payload=payload,
            target_hint=target_hint,
            mode=mode,
            adapter_name=self.adapter_name,
            execution_result=execution_result,
            default_confidence=0.7,
        )

    @classmethod
    def _blocked_template_tags(cls, metadata: dict[str, Any]) -> set[str]:
        if bool(metadata.get("allow_unsafe_nuclei_templates", False)):
            return set()
        tags = set()
        for key in ("nuclei_template_tags", "template_tags", "tags"):
            tags.update(_coerce_tags(metadata.get(key)))
        return {tag for tag in tags if tag in cls.blocked_tags}


class HttpxFingerprintAdapter(ProbeAdapter):
    """Adapter for httpx JSONL fingerprint output."""

    adapter_name = "httpx"

    def can_handle(self, *, mode: str) -> bool:
        return mode in {"service_validation", "service_validation_result", "fingerprint", "fingerprint_result"}

    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        command = [str(metadata.get("httpx_path", "httpx")), "-json", "-title", "-tech-detect", "-status-code"]
        command.extend(["-u", str(metadata.get("target_url", target_hint))])
        rate_limit = metadata.get("rate_limit") or metadata.get("httpx_rate_limit")
        if rate_limit is not None:
            command.extend(["-rate-limit", str(rate_limit)])
        return command

    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        item = next(iter(_json_lines(execution_result.stdout)), {})
        technologies = item.get("tech") if isinstance(item.get("tech"), list) else []
        status_code = item.get("status-code") or item.get("status_code")
        url = item.get("url") or item.get("input") or target_hint
        service = {
            "id": metadata.get("service_id") or str(url),
            "type": "Service",
            "host_id": metadata.get("host_id") or str(url),
            "port": metadata.get("port"),
            "protocol": "https" if str(url).startswith("https://") else "http",
            "service_name": "http",
            "banner": item.get("webserver"),
            "title": item.get("title"),
            "status_code": status_code,
            "technologies": technologies,
            "validated": bool(status_code),
        }
        payload = {
            "summary": f"httpx fingerprint completed for {url}",
            "confidence": float(metadata.get("confidence", 0.82 if item else 0.45)),
            "reachable": bool(status_code) or execution_result.success,
            "success": execution_result.success and bool(item),
            "partial_success": bool(item) and not execution_result.success,
            "service": service if item else {},
            "entities": [service] if item else [],
            "evidence": {"adapter": self.adapter_name, "fingerprint": item, "tool": execution_result.to_payload()},
            "runtime_hints": {"http_status": status_code, "technologies": technologies},
        }
        return _normalize_payload(
            payload=payload,
            target_hint=target_hint,
            mode=mode,
            adapter_name=self.adapter_name,
            execution_result=execution_result,
            default_confidence=0.7,
        )


class WhatWebFingerprintAdapter(ProbeAdapter):
    """Adapter for WhatWeb JSON fingerprint output."""

    adapter_name = "whatweb"

    def can_handle(self, *, mode: str) -> bool:
        return mode in {"service_validation", "service_validation_result", "fingerprint", "fingerprint_result"}

    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        return [str(metadata.get("whatweb_path", "whatweb")), "--log-json=-", str(metadata.get("target_url", target_hint))]

    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        decoded = _json_payload(execution_result.stdout)
        records = decoded if isinstance(decoded, list) else ([decoded] if isinstance(decoded, dict) else [])
        record = records[0] if records else {}
        plugins = record.get("plugins") if isinstance(record.get("plugins"), dict) else {}
        plugin_names = sorted(str(name) for name in plugins)
        target = record.get("target") or target_hint
        service = {
            "id": metadata.get("service_id") or str(target),
            "type": "Service",
            "host_id": metadata.get("host_id") or str(target),
            "service_name": "http",
            "plugins": plugin_names,
            "validated": bool(record),
        }
        payload = {
            "summary": f"whatweb fingerprint completed for {target}",
            "confidence": float(metadata.get("confidence", 0.78 if record else 0.45)),
            "reachable": bool(record) or execution_result.success,
            "success": execution_result.success and bool(record),
            "partial_success": bool(record) and not execution_result.success,
            "service": service if record else {},
            "entities": [service] if record else [],
            "evidence": {"adapter": self.adapter_name, "fingerprint": record, "tool": execution_result.to_payload()},
            "runtime_hints": {"whatweb_plugins": plugin_names},
        }
        return _normalize_payload(
            payload=payload,
            target_hint=target_hint,
            mode=mode,
            adapter_name=self.adapter_name,
            execution_result=execution_result,
            default_confidence=0.65,
        )


class SSLScanAdapter(ProbeAdapter):
    """Adapter for sslscan output."""

    adapter_name = "sslscan"
    _subject_re = re.compile(r"Subject:\s*(?P<subject>.+)")
    _issuer_re = re.compile(r"Issuer:\s*(?P<issuer>.+)")
    _protocol_re = re.compile(r"^\s*(?P<protocol>TLSv[0-9.]+|SSLv[0-9.]+)\s+(?P<status>enabled|disabled)", re.MULTILINE | re.IGNORECASE)

    def can_handle(self, *, mode: str) -> bool:
        return mode in {"service_validation", "service_validation_result", "fingerprint", "fingerprint_result"}

    def build_command(self, *, target_hint: str, mode: str, metadata: dict[str, Any]) -> list[str]:
        port = metadata.get("port") or metadata.get("service_port")
        target = f"{target_hint}:{port}" if port and ":" not in target_hint else target_hint
        return [str(metadata.get("sslscan_path", "sslscan")), "--no-colour", str(target)]

    def parse_output(
        self,
        *,
        execution_result: ToolExecutionResult,
        target_hint: str,
        mode: str,
        metadata: dict[str, Any],
    ) -> ParsedProbeResult:
        stdout = execution_result.stdout.strip()
        subject = _first_match(self._subject_re, stdout, "subject")
        issuer = _first_match(self._issuer_re, stdout, "issuer")
        protocols = [
            {"protocol": match.group("protocol"), "status": match.group("status").lower()}
            for match in self._protocol_re.finditer(stdout)
        ]
        service = {
            "id": metadata.get("service_id") or target_hint,
            "type": "Service",
            "host_id": metadata.get("host_id") or target_hint.split(":", 1)[0],
            "service_name": "tls",
            "tls_subject": subject,
            "tls_issuer": issuer,
            "tls_protocols": protocols,
            "validated": bool(subject or issuer or protocols),
        }
        payload = {
            "summary": f"sslscan fingerprint completed for {target_hint}",
            "confidence": float(metadata.get("confidence", 0.8 if service["validated"] else 0.45)),
            "reachable": service["validated"] or execution_result.success,
            "success": execution_result.success and service["validated"],
            "partial_success": service["validated"] and not execution_result.success,
            "service": service if service["validated"] else {},
            "entities": [service] if service["validated"] else [],
            "evidence": {
                "adapter": self.adapter_name,
                "certificate": {"subject": subject, "issuer": issuer, "protocols": protocols},
                "stdout_excerpt": stdout[:400],
                "tool": execution_result.to_payload(),
            },
            "runtime_hints": {"tls_protocols": protocols},
        }
        return _normalize_payload(
            payload=payload,
            target_hint=target_hint,
            mode=mode,
            adapter_name=self.adapter_name,
            execution_result=execution_result,
            default_confidence=0.65,
        )


def _normalize_payload(
    *,
    payload: dict[str, Any],
    target_hint: str,
    mode: str,
    adapter_name: str,
    execution_result: ToolExecutionResult,
    default_confidence: float,
) -> ParsedProbeResult:
    hosts = list(payload.get("hosts", [])) if isinstance(payload.get("hosts"), list) else []
    service = dict(payload.get("service", {})) if isinstance(payload.get("service"), dict) else {}
    identity_context = (
        dict(payload.get("identity_context", {}))
        if isinstance(payload.get("identity_context"), dict)
        else {}
    )
    entities = list(payload.get("entities", [])) if isinstance(payload.get("entities"), list) else []
    relations = list(payload.get("relations", [])) if isinstance(payload.get("relations"), list) else []
    evidence = dict(payload.get("evidence", {})) if isinstance(payload.get("evidence"), dict) else {}
    runtime_hints = dict(payload.get("runtime_hints", {})) if isinstance(payload.get("runtime_hints"), dict) else {}

    if not entities:
        if hosts:
            entities.extend({"id": item.get("host_id", target_hint), "type": "Host", **item} for item in hosts)
        if service:
            entities.append({"id": service.get("id", service.get("service_id", target_hint)), "type": "Service", **service})
        if identity_context:
            entities.append(
                {"id": identity_context.get("identity_id", target_hint), "type": "Identity", **identity_context}
            )
    if not relations and service and service.get("host_id"):
        relations.append(
            {
                "type": "HOSTS",
                "source": service["host_id"],
                "target": service.get("id", service.get("service_id", target_hint)),
                "attributes": {"port": service.get("port"), "protocol": service.get("protocol")},
            }
        )

    reachable = bool(payload.get("reachable", runtime_hints.get("reachable", execution_result.success)))
    blocked = bool(payload.get("blocked", False))
    blocked_reason = payload.get("blocked_reason")
    partial_success = bool(payload.get("partial_success", False))
    if not partial_success and not execution_result.success and not blocked:
        partial_success = bool(entities or relations or service or identity_context or hosts)
    success = bool(payload.get("success", execution_result.success and not blocked and reachable))
    failure_reason = payload.get("failure_reason")
    if failure_reason is None and not execution_result.success and not blocked:
        failure_reason = (
            execution_result.stderr.strip()
            or execution_result.error_message
            or (
                f"tool exited with code {execution_result.exit_code}"
                if execution_result.exit_code is not None
                else None
            )
        )

    if mode in {"host_discovery", "host_discovery_result"} and not hosts:
        hosts = [{"host_id": target_hint, "status": "up" if reachable else "unknown"}]
    if mode in {"service_validation", "service_validation_result"} and not service and entities:
        service = next((dict(item) for item in entities if str(item.get("type", "")).lower() == "service"), {})
    if mode in {"identity_context_discovery", "identity_context_result"} and not identity_context and entities:
        identity_context = next(
            (dict(item) for item in entities if str(item.get("type", "")).lower() == "identity"),
            {},
        )

    evidence.setdefault("adapter", adapter_name)
    evidence.setdefault("command_category", execution_result.category)
    evidence.setdefault("tool", execution_result.to_payload())
    summary = str(payload.get("summary") or f"{adapter_name} completed for {target_hint}")
    if blocked and blocked_reason:
        summary = str(payload.get("summary") or f"{adapter_name} blocked for {target_hint}: {blocked_reason}")

    return ParsedProbeResult(
        summary=summary,
        confidence=float(payload.get("confidence", default_confidence)),
        reachable=reachable,
        success=success and execution_result.success and not blocked,
        partial_success=partial_success,
        blocked=blocked,
        blocked_reason=(None if not blocked else str(blocked_reason or "tool unavailable")),
        failure_reason=(None if failure_reason is None else str(failure_reason)),
        entities=entities,
        relations=relations,
        evidence=evidence,
        runtime_hints=runtime_hints,
        hosts=hosts,
        service=service,
        identity_context=identity_context,
        raw_output=(str(payload.get("raw_output")) if payload.get("raw_output") is not None else execution_result.stdout.strip() or None),
    )


def _json_payload(value: str) -> Any:
    try:
        return json.loads(value.strip())
    except json.JSONDecodeError:
        return None


def _json_lines(value: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        decoded = _json_payload(line)
        if isinstance(decoded, dict):
            records.append(decoded)
    if not records:
        decoded = _json_payload(value)
        if isinstance(decoded, dict):
            records.append(decoded)
        elif isinstance(decoded, list):
            records.extend(item for item in decoded if isinstance(item, dict))
    return records


def _coerce_tags(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_items = re.split(r"[,;\s]+", value)
    elif isinstance(value, list | tuple | set):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    return {item.strip().lower() for item in raw_items if item and item.strip()}


def _first_match(pattern: re.Pattern[str], value: str, group_name: str) -> str | None:
    match = pattern.search(value)
    if not match:
        return None
    return match.group(group_name).strip()


__all__ = [
    "CustomProbeAdapter",
    "HttpxFingerprintAdapter",
    "MasscanAdapter",
    "NmapAdapter",
    "NucleiSafeTemplateAdapter",
    "ParsedProbeResult",
    "ProbeAdapter",
    "ProbeAdapterUnavailable",
    "SSLScanAdapter",
    "WhatWebFingerprintAdapter",
]
