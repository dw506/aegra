"""Application settings for the local control plane."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.agents.packy_llm import DEFAULT_PACKY_BASE_URL, DEFAULT_PACKY_MODEL, PackyLLMConfig, load_llm_env_file
from src.core.runtime.policy import RuntimePolicy, load_runtime_policy_payload


class AppSettings(BaseModel):
    """Environment-driven settings for orchestration and control APIs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_store_backend: Literal["memory", "file"] = "file"
    runtime_store_dir: Path = Field(default_factory=lambda: Path("var/runtime"))
    control_api_title: str = "Aegra Control API"
    control_api_version: str = "0.1.0"
    max_concurrent_workers: int = Field(default=4, ge=1)
    default_operation_budget: int = Field(default=100, ge=1)
    default_scan_timeout_sec: int = Field(default=300, ge=1)
    audit_enabled: bool = True
    audit_dir: Path = Field(default_factory=lambda: Path("var/audit"))
    audit_persist_enabled: bool = True
    audit_max_entries: int = Field(default=200, ge=1)
    operation_log_max_entries: int = Field(default=200, ge=1)
    audit_redaction_enabled: bool = True
    recovery_enabled: bool = True
    runtime_policy_path: Path | None = None
    runtime_policy: dict[str, Any] = Field(default_factory=dict)
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_sec: float | None = Field(default=None, gt=0.0, le=300.0)
    llm_input_cost_per_1m_tokens: float | None = Field(default=None, ge=0.0)
    llm_output_cost_per_1m_tokens: float | None = Field(default=None, ge=0.0)
    enable_planner_llm_advisor: bool = False
    enable_critic_llm_advisor: bool = False
    enable_supervisor_llm_advisor: bool = False
    tool_nmap_path: str = "nmap"
    tool_python_path: str = "python"
    incalmo_enabled: bool = False
    incalmo_c2_url: str | None = None
    incalmo_poll_interval_sec: float = Field(default=1.0, gt=0.0)
    incalmo_command_timeout_sec: float = Field(default=60.0, gt=0.0)

    @field_validator("runtime_store_dir", mode="before")
    @classmethod
    def _coerce_runtime_store_dir(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("audit_dir", mode="before")
    @classmethod
    def _coerce_audit_dir(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("runtime_policy_path", mode="before")
    @classmethod
    def _coerce_runtime_policy_path(cls, value: str | Path | None) -> Path | None:
        if value in {None, ""}:
            return None
        return Path(value).expanduser().resolve()

    def load_runtime_policy(self) -> RuntimePolicy:
        """Load the effective runtime policy from settings and optional external file."""

        # 中文注释：
        # settings 是 runtime policy 的唯一装载入口，统一在这里做默认值合并、
        # 文件读取、schema 校验和来源元数据填充。
        return load_runtime_policy_payload(
            inline_policy=self.runtime_policy,
            policy_path=self.runtime_policy_path,
        )

    def to_packy_llm_config(self) -> PackyLLMConfig | None:
        """Convert settings into a Packy/OpenAI-compatible client config."""

        if not self.llm_api_key:
            return None
        return PackyLLMConfig(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url or DEFAULT_PACKY_BASE_URL,
            model=self.llm_model or DEFAULT_PACKY_MODEL,
            timeout_sec=self.llm_timeout_sec or 30.0,
            input_cost_per_1m_tokens=self.llm_input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=self.llm_output_cost_per_1m_tokens,
        )

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Build settings from the current process environment."""

        if not os.getenv("AEGRA_LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            load_llm_env_file()
        environ = os.environ
        values: dict[str, object] = {}
        if "AEGRA_RUNTIME_STORE_BACKEND" in environ:
            values["runtime_store_backend"] = environ["AEGRA_RUNTIME_STORE_BACKEND"]
        if "AEGRA_RUNTIME_STORE_DIR" in environ:
            values["runtime_store_dir"] = environ["AEGRA_RUNTIME_STORE_DIR"]
        if "AEGRA_CONTROL_API_TITLE" in environ:
            values["control_api_title"] = environ["AEGRA_CONTROL_API_TITLE"]
        if "AEGRA_CONTROL_API_VERSION" in environ:
            values["control_api_version"] = environ["AEGRA_CONTROL_API_VERSION"]
        if "AEGRA_MAX_CONCURRENT_WORKERS" in environ:
            values["max_concurrent_workers"] = int(environ["AEGRA_MAX_CONCURRENT_WORKERS"])
        if "AEGRA_DEFAULT_OPERATION_BUDGET" in environ:
            values["default_operation_budget"] = int(environ["AEGRA_DEFAULT_OPERATION_BUDGET"])
        if "AEGRA_DEFAULT_SCAN_TIMEOUT_SEC" in environ:
            values["default_scan_timeout_sec"] = int(environ["AEGRA_DEFAULT_SCAN_TIMEOUT_SEC"])
        if "AEGRA_AUDIT_ENABLED" in environ:
            values["audit_enabled"] = environ["AEGRA_AUDIT_ENABLED"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_AUDIT_DIR" in environ:
            values["audit_dir"] = environ["AEGRA_AUDIT_DIR"]
        if "AEGRA_AUDIT_PERSIST_ENABLED" in environ:
            values["audit_persist_enabled"] = environ["AEGRA_AUDIT_PERSIST_ENABLED"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_AUDIT_MAX_ENTRIES" in environ:
            values["audit_max_entries"] = int(environ["AEGRA_AUDIT_MAX_ENTRIES"])
        if "AEGRA_OPERATION_LOG_MAX_ENTRIES" in environ:
            values["operation_log_max_entries"] = int(environ["AEGRA_OPERATION_LOG_MAX_ENTRIES"])
        if "AEGRA_AUDIT_REDACTION_ENABLED" in environ:
            values["audit_redaction_enabled"] = environ["AEGRA_AUDIT_REDACTION_ENABLED"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_RECOVERY_ENABLED" in environ:
            values["recovery_enabled"] = environ["AEGRA_RECOVERY_ENABLED"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_RUNTIME_POLICY_PATH" in environ:
            values["runtime_policy_path"] = environ["AEGRA_RUNTIME_POLICY_PATH"]
        if "AEGRA_RUNTIME_POLICY_JSON" in environ:
            values["runtime_policy"] = json.loads(environ["AEGRA_RUNTIME_POLICY_JSON"])
        if "AEGRA_LLM_API_KEY" in environ:
            values["llm_api_key"] = environ["AEGRA_LLM_API_KEY"] or None
        if "AEGRA_LLM_BASE_URL" in environ:
            values["llm_base_url"] = environ["AEGRA_LLM_BASE_URL"] or None
        if "AEGRA_LLM_MODEL" in environ:
            values["llm_model"] = environ["AEGRA_LLM_MODEL"] or None
        if "AEGRA_LLM_TIMEOUT_SEC" in environ:
            values["llm_timeout_sec"] = float(environ["AEGRA_LLM_TIMEOUT_SEC"])
        if "AEGRA_LLM_INPUT_COST_PER_1M_TOKENS" in environ:
            values["llm_input_cost_per_1m_tokens"] = float(environ["AEGRA_LLM_INPUT_COST_PER_1M_TOKENS"])
        if "AEGRA_LLM_OUTPUT_COST_PER_1M_TOKENS" in environ:
            values["llm_output_cost_per_1m_tokens"] = float(environ["AEGRA_LLM_OUTPUT_COST_PER_1M_TOKENS"])
        if "AEGRA_ENABLE_PLANNER_LLM_ADVISOR" in environ:
            values["enable_planner_llm_advisor"] = environ["AEGRA_ENABLE_PLANNER_LLM_ADVISOR"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_ENABLE_CRITIC_LLM_ADVISOR" in environ:
            values["enable_critic_llm_advisor"] = environ["AEGRA_ENABLE_CRITIC_LLM_ADVISOR"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_ENABLE_SUPERVISOR_LLM_ADVISOR" in environ:
            values["enable_supervisor_llm_advisor"] = environ["AEGRA_ENABLE_SUPERVISOR_LLM_ADVISOR"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_TOOL_NMAP_PATH" in environ:
            values["tool_nmap_path"] = environ["AEGRA_TOOL_NMAP_PATH"]
        if "AEGRA_TOOL_PYTHON_PATH" in environ:
            values["tool_python_path"] = environ["AEGRA_TOOL_PYTHON_PATH"]
        if "AEGRA_INCALMO_ENABLED" in environ:
            values["incalmo_enabled"] = environ["AEGRA_INCALMO_ENABLED"].strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if "AEGRA_INCALMO_C2_URL" in environ:
            values["incalmo_c2_url"] = environ["AEGRA_INCALMO_C2_URL"] or None
        if "AEGRA_INCALMO_POLL_INTERVAL_SEC" in environ:
            values["incalmo_poll_interval_sec"] = float(environ["AEGRA_INCALMO_POLL_INTERVAL_SEC"])
        if "AEGRA_INCALMO_COMMAND_TIMEOUT_SEC" in environ:
            values["incalmo_command_timeout_sec"] = float(environ["AEGRA_INCALMO_COMMAND_TIMEOUT_SEC"])
        return cls.model_validate(values)


__all__ = ["AppSettings"]
