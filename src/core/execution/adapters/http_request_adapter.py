"""Bounded HTTP request execution adapter."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult


class HttpRequestExecutionAdapter:
    """Execute scoped HTTP GET/HEAD requests for discovery workers."""

    name = "http_request"
    safe_methods = frozenset({"GET", "HEAD"})

    def supports(self, plan: ToolPlan) -> bool:
        return plan.adapter == self.name or plan.tool == self.name

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        url = str(plan.target or plan.args.get("url") or "").strip()
        method = str(plan.args.get("method") or "HEAD").upper()
        if not url:
            return self._result(plan, success=False, metadata={"category": "policy_denied", "blocked_reason": "missing_url"})
        if method not in self.safe_methods:
            return self._result(
                plan,
                success=False,
                metadata={"category": "policy_denied", "url": url, "method": method, "blocked_reason": "unsupported_method"},
            )

        same_origin = _string(plan.args.get("same_origin"))
        if same_origin and not _same_origin(url, same_origin):
            return self._result(
                plan,
                success=False,
                metadata={"category": "policy_denied", "url": url, "method": method, "blocked_reason": "cross_origin"},
            )

        timeout = _float_arg(plan.args.get("timeout_seconds"), float(plan.timeout_seconds))
        try:
            request = Request(url, method=method)
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is checked by caller-provided same-origin bounds.
                headers = dict(response.headers.items())
                metadata = {
                    "category": "success",
                    "url": url,
                    "method": method,
                    "status_code": response.status,
                    "headers": headers,
                    "content_type": response.headers.get("content-type"),
                    "auth_required": response.status in {401, 403},
                    "reachable": 200 <= response.status < 500,
                }
                return self._result(plan, success=True, exit_code=response.status, metadata=metadata)
        except HTTPError as exc:
            headers = dict(exc.headers.items()) if exc.headers else {}
            metadata = {
                "category": "http_error",
                "url": url,
                "method": method,
                "status_code": exc.code,
                "headers": headers,
                "content_type": headers.get("Content-Type") or headers.get("content-type"),
                "auth_required": exc.code in {401, 403},
                "reachable": 200 <= exc.code < 500,
                "error_message": str(exc),
            }
            return self._result(plan, success=metadata["reachable"], exit_code=exc.code, stderr=str(exc), metadata=metadata)
        except TimeoutError as exc:
            return self._result(
                plan,
                success=False,
                exit_code="timeout",
                stderr=str(exc),
                metadata={"category": "timeout", "url": url, "method": method, "reachable": False, "error_message": str(exc)},
            )
        except URLError as exc:
            message = str(exc.reason if hasattr(exc, "reason") else exc)
            return self._result(
                plan,
                success=False,
                exit_code="request_error",
                stderr=message,
                metadata={"category": "request_error", "url": url, "method": method, "reachable": False, "error_message": message},
            )
        except Exception as exc:
            return self._result(
                plan,
                success=False,
                exit_code="request_error",
                stderr=str(exc),
                metadata={"category": "request_error", "url": url, "method": method, "reachable": False, "error_message": str(exc)},
            )

    def _result(
        self,
        plan: ToolPlan,
        *,
        success: bool,
        exit_code: int | str | None = None,
        stderr: str = "",
        metadata: dict[str, Any],
    ) -> ToolExecutionResult:
        stdout = json.dumps(metadata, sort_keys=True)
        return ToolExecutionResult(
            adapter=self.name,
            tool=plan.tool,
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metadata={"tool_plan": plan.model_dump(mode="json"), **metadata},
        )


def _same_origin(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    base = urlparse(base_url)
    return parsed.scheme == base.scheme and parsed.netloc == base.netloc


def _string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_arg(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["HttpRequestExecutionAdapter"]
