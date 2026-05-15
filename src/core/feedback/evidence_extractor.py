"""Extract standard evidence records from controlled probe output."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


class EvidenceExtractor:
    """Convert tool output into compact KG-ready evidence payloads."""

    def extract(self, result: Any) -> list[dict[str, Any]]:
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result or {})
        output = payload.get("output") if isinstance(payload.get("output"), dict) else payload
        evidence: list[dict[str, Any]] = []
        target = str(payload.get("target") or output.get("target") or "")
        host = str(output.get("host") or _host_from_target(target) or "")
        port = _int_or_none(output.get("port") or _port_from_target(target))

        if output.get("reachable"):
            evidence.append({"type": "host_reachable", "host": host, "confidence": 0.9})

        service_name = output.get("service_name") or _service_name(output)
        if host and port and service_name:
            evidence.append(
                {
                    "type": "service_detected",
                    "host": host,
                    "port": port,
                    "service_name": service_name,
                    "confidence": float(output.get("confidence") or 0.9),
                }
            )

        if output.get("http_status") is not None and host and port:
            evidence.append(
                {
                    "type": "http_status",
                    "host": host,
                    "port": port,
                    "status": int(output["http_status"]),
                    "confidence": 0.9,
                }
            )

        banner = output.get("banner")
        title = output.get("title")
        if (banner or title) and host and port:
            evidence.append(
                {
                    "type": "banner",
                    "host": host,
                    "port": port,
                    "banner": banner,
                    "title": title,
                    "confidence": 0.8,
                }
            )

        for entity in output.get("entities", []):
            if isinstance(entity, dict) and entity.get("port"):
                evidence.append(
                    {
                        "type": "open_port",
                        "host": str(entity.get("host_id") or host),
                        "port": int(entity["port"]),
                        "protocol": str(entity.get("protocol") or "tcp"),
                        "confidence": float(output.get("confidence") or 0.85),
                    }
                )

        raw_output = str(output.get("raw_output") or "")
        evidence.extend(_extract_nmap_evidence(raw_output, fallback_host=host))
        return _dedupe(evidence)


def _extract_nmap_evidence(raw_output: str, *, fallback_host: str) -> list[dict[str, Any]]:
    if not raw_output:
        return []
    host = fallback_host
    report = re.search(r"^Nmap scan report for (?P<host>.+)$", raw_output, flags=re.MULTILINE)
    if report:
        host = report.group("host").strip()
    result: list[dict[str, Any]] = []
    if "Host is up" in raw_output and host:
        result.append({"type": "host_reachable", "host": host, "confidence": 0.8})
    for match in re.finditer(
        r"^(?P<port>\d+)/(?P<protocol>tcp|udp)\s+open\s+(?P<service>\S+)(?:\s+(?P<banner>.+))?$",
        raw_output,
        flags=re.MULTILINE,
    ):
        result.append(
            {
                "type": "service_detected",
                "host": host,
                "port": int(match.group("port")),
                "service_name": match.group("service"),
                "confidence": 0.85,
            }
        )
        result.append(
            {
                "type": "open_port",
                "host": host,
                "port": int(match.group("port")),
                "protocol": match.group("protocol"),
                "confidence": 0.85,
            }
        )
        if match.group("banner"):
            result.append(
                {
                    "type": "banner",
                    "host": host,
                    "port": int(match.group("port")),
                    "banner": match.group("banner").strip(),
                    "confidence": 0.75,
                }
            )
    return result


def _service_name(output: dict[str, Any]) -> str | None:
    for entity in output.get("entities", []):
        if isinstance(entity, dict) and entity.get("service_name"):
            return str(entity["service_name"])
    return None


def _host_from_target(target: str) -> str | None:
    parsed = urlparse(target)
    return parsed.hostname


def _port_from_target(target: str) -> int | None:
    parsed = urlparse(target)
    if parsed.port:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = tuple(sorted((str(k), str(v)) for k, v in item.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


__all__ = ["EvidenceExtractor"]
