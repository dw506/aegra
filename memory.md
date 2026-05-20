# Aegra 重构记忆

## 本次计划落地记录

日期：2026-05-20

目标：将 Aegra 收敛到 `src/app`、`src/core`、`src/integrations/incalmo` 的生产主干结构，并把 Incalmo C2 作为外部执行适配器接入，而不是内置 Incalmo 的 Flask/Celery C2 server。

## 已完成改动

- API 入口由单文件 `src/app/api.py` 改为 package facade：`src/app/api/__init__.py`。
- 保留兼容入口：`from src.app import api`、`api.create_app()`、`api.app`。
- 新增 API 路由边界模块：
  - `src/app/api/operation_routes.py`
  - `src/app/api/graph_routes.py`
  - `src/app/api/execution_routes.py`
- `AppSettings` 新增 Incalmo 配置：
  - `incalmo_enabled`
  - `incalmo_c2_url`
  - `incalmo_poll_interval_sec`
  - `incalmo_command_timeout_sec`
- 新增示例配置：`configs/incalmo.yaml`。
- 新增 `src/core/actions/`：
  - 定义 action template schema。
  - 实现 `enumerate_host`、`validate_service`、`validate_reachability`、`establish_pivot`、`reuse_credential`、`validate_goal` 到 TG task 的确定性映射。
- 新增 `src/core/execution/`：
  - `ToolPlan`
  - `ToolPolicy`
  - `ToolExecutor`
  - `ExecutionAdapter` 协议
  - `IncalmoC2Adapter`
- 新增 `src/integrations/incalmo/`：
  - `models.py`：Incalmo Agent / Command / CommandStatus / CommandResult / LLM action payload。
  - `client.py`：Incalmo C2 HTTP client。
  - `mapper.py`：Incalmo 对象与 Aegra runtime/session/evidence/result 的映射。
- 新增 `src/core/perception/c2_parser.py`：
  - 将 Incalmo command output 转成 observations 和 fact write requests。
  - adapter 不直接写 KG。
- 新增 worker 收敛命名兼容模块：
  - `service_validation_worker.py`
  - `web_enum_worker.py`
  - `credential_worker.py`
  - `pivot_worker.py`
- 清理删除：
  - `benchmarks/`
  - `scratch_*.py`
  - `tmp-*`
  - `tmp-report.*`
  - `compare*`
  - `AutoPentester.txt`
  - `PentestGPT_paper.txt`
  - 顶层旧 `kg/`
  - benchmark / 旧 `kg` / scratch 相关测试与 fixture
- 更新 `.gitignore`，忽略 benchmark logs、scratch 文件、tmp report、运行缓存。

## 保持不变的关键边界

- `AppOrchestrator` 仍是应用层唯一调度入口。
- 主循环仍保持 RuntimeStore / GraphMemoryStore / AgentPipeline / ResultApplier 的所有权边界。
- Incalmo 仅作为外部 C2 通信适配器。
- session、pivot、KG/AG/TG 更新仍由 Aegra runtime managers 和 `ResultApplier` 负责。
- 远程命令执行必须经过 Aegra scope、budget、policy gate、approval gate。
- `incalmo_enabled` 默认关闭，未配置 C2 时不改变当前本地执行行为。

## 测试结果

已运行：

```powershell
python -m pytest -q
```

结果：

```text
359 passed, 4 skipped
```

同时检查：

```powershell
rg "from kg|import kg|scratch_packy|benchmarks" src tests
```

结果：无匹配，生产代码和测试不再依赖已删除的 benchmark、scratch 和旧顶层 `kg` 包。

## 当前代码目录

```text
configs/
  incalmo.yaml

src/
  app/
    api/
      __init__.py
      execution_routes.py
      graph_routes.py
      operation_routes.py
    static/
      app.js
      index.html
      styles.css
    orchestrator.py
    settings.py

  core/
    actions/
      __init__.py
      schemas.py
    agents/
      agent_models.py
      agent_pipeline.py
      agent_protocol.py
      critic.py
      graph_context.py
      graph_llm_models.py
      graph_llm_planner.py
      graph_projection.py
      kg_events.py
      llm_decision.py
      packy_critic_advisor.py
      packy_llm.py
      packy_planner_advisor.py
      packy_supervisor_advisor.py
      perception.py
      pipeline_builders.py
      planner.py
      registry.py
      scheduler_agent.py
      state_writer.py
      supervisor.py
      task_builder.py
    execution/
      adapters/
        __init__.py
        incalmo_c2_adapter.py
      __init__.py
      executor.py
      tool_plan.py
      tool_policy.py
    feedback/
      __init__.py
      evidence_extractor.py
      result_verifier.py
    graph/
      ag_projector.py
      graph_initializer.py
      graph_memory_store.py
      kg_store.py
      tg_builder.py
      tg_merge.py
    ingestion/
      perception.py
      state_writer.py
    models/
      ag.py
      events.py
      finding.py
      fingerprint.py
      kg.py
      kg_enums.py
      kg_exceptions.py
      kg_query.py
      kg_types.py
      runtime.py
      scope.py
      tg.py
      vulnerability_candidate.py
    perception/
      __init__.py
      c2_parser.py
    planner/
      critic.py
      planner.py
      scorer.py
    runtime/
      approvals.py
      audit_report.py
      budgets.py
      checkpoint_store.py
      credential_manager.py
      events.py
      lease_manager.py
      llm_history.py
      locks.py
      observability.py
      pivot_route_manager.py
      policy.py
      policy_engine.py
      policy_gate.py
      reducer.py
      repetition_detector.py
      report_generator.py
      result_applier.py
      risk_scoring.py
      runtime_queries.py
      scheduler.py
      session_manager.py
      store.py
    tools/
      __init__.py
      recipe.py
      registry.py
      runner.py
    vuln_candidates/
      __init__.py
      matcher.py
      rules.py
    workers/
      vulnerability_validators/
        __init__.py
        base.py
        default_credentials_idor.py
        http_fingerprint.py
        registry.py
        struts2_s2045.py
      access_validation_worker.py
      access_validators.py
      access_worker.py
      base.py
      credential_worker.py
      fingerprint_worker.py
      general_worker.py
      goal_command_validator.py
      goal_validation_worker.py
      goal_validator.py
      goal_worker.py
      pivot_worker.py
      privilege_validation_worker.py
      probe_adapters.py
      recon_worker.py
      service_validation_worker.py
      tool_runner.py
      vulnerability_validation_worker.py
      web_enum_worker.py

  integrations/
    __init__.py
    incalmo/
      __init__.py
      client.py
      mapper.py
      models.py

tests/
  test_action_templates.py
  test_ag_models.py
  test_ag_projector.py
  test_agent_workers.py
  test_agents.py
  test_api.py
  test_api_operation_cycle.py
  test_app_orchestrator.py
  test_approval_api.py
  test_audit_report.py
  test_candidate_to_task_graph.py
  test_critic.py
  test_findings.py
  test_findings_api.py
  test_fingerprint_adapters.py
  test_fingerprint_worker.py
  test_graph_context.py
  test_graph_initializer.py
  test_graph_iteration_vuln_env_smoke.py
  test_graph_llm_models.py
  test_graph_llm_planner.py
  test_graph_memory_store.py
  test_incalmo_integration.py
  test_incalmo_settings.py
  test_llm_decision_history.py
  test_llm_decision_models.py
  test_nuclei_safe_adapter.py
  test_packy_critic_advisor.py
  test_packy_llm.py
  test_packy_planner_advisor.py
  test_phase_two_result_applier.py
  test_phase_two_workers.py
  test_pipeline_builders.py
  test_planner.py
  test_policy_gate.py
  test_policy_scheduler_gate.py
  test_product_api_ui.py
  test_repetition_detector.py
  test_report_generator.py
  test_risk_scoring.py
  test_runtime_budgets.py
  test_runtime_checkpoint_store.py
  test_runtime_credentials.py
  test_runtime_events.py
  test_runtime_leases.py
  test_runtime_locks.py
  test_runtime_pivot_routes.py
  test_runtime_scheduler.py
  test_runtime_sessions.py
  test_runtime_state.py
  test_runtime_store.py
  test_scope_policy.py
  test_scorer.py
  test_supervisor_agent.py
  test_tg_builder.py
  test_tg_lifecycle_progression.py
  test_tg_merge.py
  test_tg_models.py
  test_tool_execution.py
  test_tool_policy.py
  test_tool_recipe_feedback.py
  test_vulhub_kg_smoke.py
  test_vulhub_orchestrator_smoke.py
  test_vulnerability_candidate_matcher.py
  test_vulnerability_validation_worker.py
  test_vulnerability_validator_models.py
  test_vulnerability_validator_registry.py
```
