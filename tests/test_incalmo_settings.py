from __future__ import annotations

from src.app.settings import AppSettings


def test_incalmo_settings_defaults_disabled() -> None:
    settings = AppSettings()

    assert settings.mcp_enabled is False
    assert settings.mcp_first is True
    assert settings.mcp_config_path is None
    assert settings.mcp_config_json == {}
    assert settings.mcp_default_timeout_seconds == 60
    assert settings.allow_local_fallback is True
    assert settings.incalmo_enabled is False
    assert settings.incalmo_c2_url is None
    assert settings.incalmo_poll_interval_sec == 1.0
    assert settings.incalmo_command_timeout_sec == 60.0


def test_incalmo_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_INCALMO_ENABLED", "true")
    monkeypatch.setenv("AEGRA_INCALMO_C2_URL", "http://c2.local")
    monkeypatch.setenv("AEGRA_INCALMO_POLL_INTERVAL_SEC", "0.25")
    monkeypatch.setenv("AEGRA_INCALMO_COMMAND_TIMEOUT_SEC", "12.5")

    settings = AppSettings.from_env()

    assert settings.incalmo_enabled is True
    assert settings.incalmo_c2_url == "http://c2.local"
    assert settings.incalmo_poll_interval_sec == 0.25
    assert settings.incalmo_command_timeout_sec == 12.5


def test_mcp_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_MCP_ENABLED", "true")
    monkeypatch.setenv("AEGRA_MCP_FIRST", "false")
    monkeypatch.setenv("AEGRA_MCP_CONFIG_PATH", "configs/mcp.json")
    monkeypatch.setenv("AEGRA_MCP_CONFIG_JSON", '{"servers":{"pentest-tools":{}}}')
    monkeypatch.setenv("AEGRA_MCP_DEFAULT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("AEGRA_ALLOW_LOCAL_FALLBACK", "false")

    settings = AppSettings.from_env()

    assert settings.mcp_enabled is True
    assert settings.mcp_first is False
    assert settings.mcp_config_path is not None
    assert settings.mcp_config_path.name == "mcp.json"
    assert settings.mcp_config_json == {"servers": {"pentest-tools": {}}}
    assert settings.mcp_default_timeout_seconds == 45
    assert settings.allow_local_fallback is False
