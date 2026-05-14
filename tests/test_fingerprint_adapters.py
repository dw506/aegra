from __future__ import annotations

from src.core.workers.probe_adapters import (
    HttpxFingerprintAdapter,
    SSLScanAdapter,
    WhatWebFingerprintAdapter,
)
from src.core.workers.tool_runner import ToolExecutionResult


def test_httpx_fingerprint_output_normalizes_to_service_evidence() -> None:
    adapter = HttpxFingerprintAdapter()

    parsed = adapter.parse_output(
        execution_result=ToolExecutionResult(
            command=["httpx", "-json"],
            success=True,
            category="success",
            exit_code=0,
            stdout='{"url":"https://example.test","status-code":200,"title":"Example","tech":["nginx"]}\n',
        ),
        target_hint="https://example.test",
        mode="service_validation",
        metadata={},
    )

    assert parsed.success is True
    assert parsed.service["status_code"] == 200
    assert parsed.service["technologies"] == ["nginx"]
    assert parsed.evidence["adapter"] == "httpx"


def test_whatweb_fingerprint_output_normalizes_plugins() -> None:
    adapter = WhatWebFingerprintAdapter()

    parsed = adapter.parse_output(
        execution_result=ToolExecutionResult(
            command=["whatweb", "--log-json=-"],
            success=True,
            category="success",
            exit_code=0,
            stdout='[{"target":"https://example.test","plugins":{"nginx":{},"HTTPServer":{}}}]',
        ),
        target_hint="https://example.test",
        mode="service_validation",
        metadata={},
    )

    assert parsed.success is True
    assert parsed.service["plugins"] == ["HTTPServer", "nginx"]
    assert parsed.evidence["adapter"] == "whatweb"


def test_sslscan_output_normalizes_certificate_evidence() -> None:
    adapter = SSLScanAdapter()

    parsed = adapter.parse_output(
        execution_result=ToolExecutionResult(
            command=["sslscan", "example.test:443"],
            success=True,
            category="success",
            exit_code=0,
            stdout=(
                "Subject:  example.test\n"
                "Issuer:   Example CA\n"
                "  TLSv1.2 enabled\n"
                "  TLSv1.3 enabled\n"
            ),
        ),
        target_hint="example.test:443",
        mode="service_validation",
        metadata={},
    )

    assert parsed.success is True
    assert parsed.service["tls_subject"] == "example.test"
    assert parsed.evidence["certificate"]["issuer"] == "Example CA"
    assert parsed.runtime_hints["tls_protocols"][0]["protocol"] == "TLSv1.2"
