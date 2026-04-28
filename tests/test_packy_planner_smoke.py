from __future__ import annotations

from types import SimpleNamespace

import pytest

import scratch_packy_planner_smoke


def test_smoke_main_exits_with_clear_message_when_llm_env_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGRA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("AEGRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AEGRA_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        scratch_packy_planner_smoke.main()

    assert "缺少 LLM 环境变量" in str(exc_info.value)


def test_smoke_main_prints_llm_advice_summary_when_advice_exists(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def run_planning_cycle(self, **kwargs):  # noqa: ANN003
            del kwargs
            return SimpleNamespace(
                success=True,
                errors=[],
                final_output=SimpleNamespace(
                    decisions=[
                        {
                            "summary": "Selected planning candidate #1 for goal goal-1",
                            "score": 0.91,
                            "rationale": "baseline; llm 建议该候选更适合当前目标",
                            "payload": {
                                "planning_candidate": {
                                    "action_ids": ["action-1"],
                                    "metadata": {
                                        "llm_advice": {
                                            "candidate_id": "cand-1",
                                            "score_delta": 0.1,
                                            "metadata": {"reason": "goal_alignment"},
                                        }
                                    },
                                }
                            },
                        }
                    ],
                ),
                logs=["planner ran"],
            )

    monkeypatch.setattr(scratch_packy_planner_smoke, "build_packy_planner_pipeline", lambda: FakePipeline())

    scratch_packy_planner_smoke.main()
    captured = capsys.readouterr()

    assert "Packy Planner Smoke" in captured.out
    assert "goal_alignment" in captured.out
