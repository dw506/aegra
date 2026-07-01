# Aegra 代码地图（ml.md）

> 自主多主机渗透框架，PEV（Plan-Execute-Verify）循环 + 3 张图（KG/AG/Runtime）。
> 本文档基于当前代码（v3 + LangGraph 全落地后）生成：先讲整体流程与数据转换，再逐文件说明每个代码块的职责。

---

## 一、整体流程

### 1.1 Operation 生命周期
```
create_operation → import_targets → start_operation → run_until_quiescent(循环控制周期直到 stop/静默)
                                          │
                                          └─ GraphInitializer 种子化 KG：
                                             Host + NetworkZone(scope) + Goal + BELONGS_TO_ZONE/TARGETS 边
```

### 1.2 一个控制周期 —— LangGraph `AppOrchestrator._build_operation_cycle_graph`（6 节点）
```
load_context → prepare_planner_context → planner_decide ─┬─(execute)→ apply_planner_outcome ─┬→ execute_round → finalize_cycle → END
  快照 Runtime       build_min_summary        Planner.decide  └─(stop/pause/replan)────────────┴→ finalize_cycle
  载入 KG/AG         + policy + recent          (产 PlannerOutcome)
```

### 1.3 PEV 三段（每段各自也是 LangGraph）

**① Planner**（`planner_loop.run_planner_loop`，decide/act 循环，push 模型）
- 种子：`min_summary`（KG/AG 计数摘要）+ `success_condition_progress` + policy + recent_outcomes
- 每轮：LLM 要么调图**读工具**（kg_query 等）钻取，要么吐**最终 PlannerOutcome**
- 产出：`PlannerOutcome{action: execute|replan|pause_for_review|stop_success|stop_failed, directive}`
- `RoundDirective{objective, target_refs, allowed_tools, tool_hints, max_tools, success_hint, risk_level}` —— **无 capability**（已删，相模型残留）

**② Executor**（`execution_agent._build_execution_graph`，bounded 工具循环）
- `RoundDirective → ExecutionRequest`，循环至 `max_steps`
- 每轮：LLM 吐 `call_mcp_tool | finish | need_replan`；工具经 MCP client → `ToolTrace`
- 护栏：no-progress guard（连续 3 次无进展早停）、transport 上下文（session/route 盖到工具参数）
- **能力无关**：只从 objective + success_criteria 推理
- 产出：`ExecutionResult{status, summary, evidence_refs, tool_trace[], runtime_hints, ...}` —— **无 channel-② 自报**

**③ ResultApplier**（`PhaseTwoResultApplier.apply_execution_result`，全 channel ①）
```
tool_trace ─┬→ ToolTraceFactExtractor → ExtractedFact → patch delta → kg_store.apply_patch_batch → KG 节点
            ├→ _harvest_runtime_facts(读 parsed_output.runtime_hints) → RuntimeState(session/route/credential)
            └→ _record_attack_step → AG：1 个 ATTACK_STEP + NEXT 边  + audit log
```

### 1.4 成功判定 —— `SuccessConditionTracker`（声明式、确定性）
读 `success_contract.yml` 的谓词（`exists_node`+`filters:{kind}`、`service_discovered_via_route` 等）对**活的 KG/AG/Runtime** 求值，产 `success_condition_progress{missing, eligible_for_stop, achieved_level}`。**Planner 读裁决，不凭感觉**——`eligible_for_stop=true` 才允许 `stop_success`。

### 1.5 核心不变量
Planner 拥有全部判断（选目标/停/成功）；Executor 只管"这轮目标达没达"；成功=声明式合约确定性求值；写图不阻断控制流；机器事实全走 channel ①（工具=权威）；秘密永不进图（GoalOracle HMAC/proof_token）。

---

## 二、工具数据转换链（5 跳）

```
外部 MCP 工具(JSON-RPC) 
  ─①configured_mcp_client._tool_result_from_rpc→ MCPToolCallResult{success,content,stdout,stderr,exit_code,metadata:{server_id,parsed_output}}
  ─②execution_agent._call_mcp_tool→ ToolTrace{tool_name,args,success,summary,stdout,parsed_output(dict),raw_output_ref(URI),metadata}
  ─③ToolTraceFactExtractor→ ExtractedFact{entity_type,label,properties{kind},confidence,source_tool,zone_ref}
  ─④result_applier._fact_deltas→ patch delta{id,payload:{patch_kind},patch:{entity_kind,entity_id,entity_type,attributes,confidence}}
  ─⑤kg_store.apply_patch_batch→ 类型化 KG 节点(Host/Service/Evidence{kind}/Session/PivotRoute/...)
```

三种载荷分工（非重复）：`stdout`=文本(正则型 extractor)、`parsed_output`=结构化 dict(字段型 extractor + runtime_hints harvest)、`raw_output_ref`=URI 指针(evidence payload_ref)。

---

## 三、三张图（状态）

| 图 | 职责 | 规模 |
|---|---|---|
| **KG** `KnowledgeGraph` | 世界模型真相 | 9 NodeType / 4 EdgeType；Evidence{kind} 承载塌缩类型(vuln_candidate/exploit_attempt/post_access/credential/controlled_read/goal_check) |
| **AG** `AttackGraph` | 结果叙事 | 1 NodeType(ATTACK_STEP)/ 1 EdgeType(NEXT) |
| **Runtime** `RuntimeState` | 会话/凭证/路由/预算/近期结果/replan | lean（无事件溯源） |

**持久化**：`RuntimeStore`（只存快照）+ `GraphMemoryStore`（KG/AG/Runtime 快照）。恢复=规整最新快照。

---

## 四、目录逐文件说明

### `src/app/` —— 应用层（控制平面入口）

**`orchestrator.py`** — 顶层编排入口
- `AppOrchestrator` — 唯一控制平面对象。`create_operation`/`import_targets`/`start_operation` 生命周期；`_build_operation_cycle_graph` 编译 6 节点 LangGraph；`_operation_*_node` 六个节点实现（load_context/prepare_planner_context/planner_decide/apply_planner_outcome/execute_round/finalize_cycle）+ `_operation_after_*` 条件边；`run_operation_cycle`/`run_until_quiescent` 驱动；`_update_success_condition_progress` 跑 tracker；`list_findings`/`get_findings_graph`/`export_findings_report` 报告；`get_health/readiness_status` 健康检查。
- `OperationGraphState` — 控制周期的 LangGraph state。
- `OperationSummary`/`OperationCycleResult`/`OperationRunSummary` — 对外汇总模型。
- `TargetHost` — 导入目标输入模型。

**`api/__init__.py`** — REST API 表面
- `create_app()` — 构建 FastAPI 应用，挂 operations/findings/graph/health 路由。
- `*Request` 一组 — 请求体模型（创建/导入/动作/周期/运行）。
- `_default_graph_refs()` — 默认 kg-root/ag-root 引用；`_run_operation_response`/`_workspace_from_summary`/`_assets_from_state` 响应装配。

**`settings.py`** — 配置
- `AppSettings` — 运行时存储后端/目录、runtime_policy、lab_profile、LLM 配置的加载与规整；`to_packy_llm_config()` 产 LLM 客户端配置；`from_env()` 环境变量装配。

### `src/core/planning/` —— 规划（thick agentic planner）

**`planner.py`** — planner agent（bounded LLM 工具循环）
- `Planner.decide()` — orchestrator 入口，建 loop state → 跑 `run_planner_loop` → 应用 advisory 写工具调用。
- `Planner.run_turn()` — 一个 LLM turn：要么返回读工具调用，要么解析出最终 `PlannerOutcome`（失败降级 fallback）。
- `_build_turn_prompt()` — 组 PlannerOutcome contract + 上下文；`_slim_tool_catalog` 瘦身 catalog（去 inputSchema）；`_fallback_outcome` 降级 outcome。
- `SYSTEM_PROMPT` — 载入 `prompts/planner_global_control.md`。

**`planner_loop.py`** — LangGraph planner 循环
- `PlannerLoopState` — 一个决策序列的 typed state。
- `decide_node`/`act_node` — 纯 `State→State` 节点（decide 调 planner.run_turn，act 执行读工具）。
- `build_planner_graph`/`run_planner_loop` — 编译 + 驱动 StateGraph 直到出 outcome 或读预算耗尽。

**`graph_tools.py`** — planner 的 typed 图工具面
- `PlannerGraphTools` — 读工具：`build_min_summary`（KG/AG 计数摘要，注入每轮）、`query_kg_nodes`/`get_node`/`get_attack_steps`/`list_runtime`；写工具（advisory 判断记录）：`record_finding`(产 Finding 节点)/`link_evidence`(产 SUPPORTED_BY 边)/`record_attack_step`(仅记 intent，AG 节点由 ResultApplier 拥有)；`tool_manifest`/`apply_tool_calls` 分发。
- `RecordFindingRequest`/`LinkEvidenceRequest`/`RecordAttackStepRequest` — 写工具入参模型。

**`models.py`** — `PlannerOutcome`（action + directive|None + reason/stop_condition/confidence/metadata；`validate_action_payload` 强制 execute↔directive 非空、stop/replan↔directive 空）。

### `src/core/execution/` —— 执行（thin single executor）

**`execution_agent.py`** — 唯一执行 agent
- `ExecutionAgent` — 公共入口：`run(RoundDirective|ExecutionRequest)`，`RoundDirective→ExecutionRequest`（不再传 capability），委托内部 loop。
- `_ExecutionLoop` — bounded 工具循环，编译成 LangGraph：`_build_execution_graph` + 节点 `_execution_start_node`/`_execution_decide_node`(LLM 决策)/`_execution_call_tool_node`/`_execution_partial_result_node` + `_execution_after_*` 条件边；`_call_mcp_tool`（MCPToolCallResult→ToolTrace，含 scope 硬闸/catalog 校验/transport 盖参）；`_finish_result`/`_build_execution_result_from_finish`（finish payload→ExecutionResult，`_empty_success_needs_replan` 判空成功）；`_replan_result`/`_partial_result_from_tool_memory`；`_tool_call_signature` no-progress 去重；`_build_messages`/`_call_llm`。
- `_ExecutionLoopState` — 执行轮的 LangGraph state。
- `_ExecutionToolCall` — 一次工具调用的 typed 描述。

**`models.py`** — 执行契约
- `RoundDirective` — planner→executor 合约（objective 导向，无 capability；`_coerce_target_refs`/`_coerce_tool_hints` 容 LLM 漂移；`_drop_legacy_capability` before-validator 丢弃残留 capability）。
- `ExecutionRequest` — 由 directive 派生、喂给执行 agent。
- `ExecutionResult` — 执行输出（status/summary/evidence_refs/tool_trace/runtime_hints/…；`normalize_execution_result_payload` 解 finish 包装 + 丢 capability；**无 channel-② 自报**）。
- `ToolTrace` — 一次工具调用的审计记录（stdout/parsed_output/raw_output_ref/success/…）。

**`mcp_client.py`** — MCP 客户端协议
- `MCPToolCallResult` — 归一化的工具响应；`MCPClient`(Protocol) `list_tools`/`call_tool`；`UnavailableMCPClient` 无 MCP 时的降级。

**`configured_mcp_client.py`** — 真 MCP 传输
- `ConfiguredMCPClient` — 配置驱动的客户端：`call_tool`→`_call_rpc`/`_call_http`/`_call_stdio`；`_tool_result_from_rpc`（JSON-RPC→MCPToolCallResult，含 `_structured_content`/`_payload_field` 解包，已删 raw_mcp 冗余）；`_http_opener_for_url` 本地绕代理。
- `_StdioMCPSession` — stdio 传输的持久会话（initialize 握手/请求/读 stdout·stderr）。
- `MCPServerConfig`/`MCPRuntimeConfig` — 服务器/运行时配置。

### `src/core/runtime/` —— 运行时（写回 + 状态 + 观测）

**`result_applier.py`** — 唯一 v3 写回owner
- `PhaseTwoResultApplier` — `apply_execution_result`：channel ① 全流程。`_harvest_runtime_facts`(tool_trace.runtime_hints→session/route/credential)；`_apply_runtime`/`_record_runtime_metadata`(写 RuntimeState + 一条 runtime_state_updated audit)；`_fact_deltas`(三源[extractor facts/evidence_refs/goal-proof]直接产 patch delta，无中间 record 层，Service 内联合成 HOSTS)；`_entity_delta`/`_relation_delta`(建 patch dict)；`_record_attack_step`(AG 一步 + NEXT 边)；`_audit`。`apply_planner_outcome` 记 planner 决策。
- `PhaseTwoApplyResult` — 写回结果聚合（kg_state_deltas/runtime_updates/ag_graph/diagnostics）。

**`tool_trace_fact_extractor.py`** — 确定性事实提取器（channel ①）
- `ToolTraceFactExtractor.extract`/`extract_all` — 按 tool_name 分发 `_TOOL_EXTRACTORS`，无匹配走 generic fallback→`Evidence{tool_name}`；只处理 success=True。
- `_extract_*` 一组 — 各工具的提取：`_extract_nmap_scan`/`_extract_run_command`/`_internal_services_from_nmap_stdout`(stdout 正则→Host/Service)、`_extract_nuclei_scan`(→Evidence{kind:vuln_candidate})、`_extract_http_probe`/`_extract_web_fingerprint`、`_extract_msf`(metasploit_exec→Session+Evidence{kind:exploit_attempt})、`_extract_session`、`_extract_pivot_route`/`_extract_pivot_exec`(→PivotRoute+内网 Service)、`_extract_post_access`(→Evidence{kind:post_access})、`_extract_goal_check`(→GoalProof+Evidence{kind:goal_check})、`_extract_controlled_data_read`(→Evidence{kind:controlled_read})。
- `ExtractedFact`/`FactExtractionResult` — 提取结果模型。

**`session_manager.py`** / **`credential_manager.py`** / **`pivot_route_manager.py`** — 三个 Runtime 事实管理器
- `RuntimeSessionManager` — `open_session`/`bind_execution_to_session`/`is_session_usable`（会话开/绑/租约）。
- `RuntimeCredentialManager` — `upsert_credential`/`record_validation`/`mark_valid/invalid/expired/revoked`/`bind_target`。
- `RuntimePivotRouteManager` — `register_candidate`/`activate_route`（跳板路由注册/激活）。

**`store.py`** — Runtime 状态持久化（只存快照，无事件溯源）
- `RuntimeStore`(ABC) — `get_state`/`save_state`/`snapshot`/`create_operation`/`recover_operation`/`export_*`。
- `InMemoryRuntimeStore`/`FileRuntimeStore` — 内存 / 文件后端实现。

**`observability.py`** — 观测 + 恢复
- `append_audit_log`/`append_operation_log`/`record_phase_checkpoint` 写日志；`prepare_state_for_resume`（规整最新快照 + 清 legacy replay metadata）；`build_recovery_snapshot`/`mark_clean/unclean_shutdown`；一组 `_sanitize_*`/`_redact_*` 脱敏。

**`policy.py`** — 运行时策略
- `RuntimePolicy` — authorized_hosts/blocked_hosts/risk_policy 等的 schema + 规整 + `to_runtime_metadata`；`PolicyDecision`；`load_runtime_policy_payload`/`policy_from_runtime_state`。

**`report_generator.py`** — 报告
- `ReportGenerator` — 从 `state.execution.metadata` 建 findings/evidence/audit 报告；`export`(json/csv/md)、`graph`(findings 溯源图)、`_sanitize`/`_redact_inline` 脱敏。

**`audit_report.py`** — operation 级审计报告装配（LLM/控制可观测性），一组纯函数 `build_operation_audit_report` + `_build_correlations`/`_budget_summary`/`_sanitize` 等。

**`txt_trace_logger.py`** — 追加式分类文本轨迹
- `TxtTraceLogger` — `operation_trace`(每 op 一份 canonical 轨迹)/`write_header`/`write_block`（planner/tool/execution 各块），带脱敏。

### `src/core/graph/` —— 图存储

**`kg_store.py`** — 内存知识图
- `KnowledgeGraph` — `apply_patch_batch`(逐 delta 容错写)；`_apply_entity_patch`/`_apply_relation_patch`(patch→类型化节点/边，`_normalize_node_type`/`_normalize_edge_type` 别名、`_split_patch_attributes` 模型字段 vs properties 溢出)；`add/update/get/list_nodes`/`_edges`；`_validate_edge_constraints`(如 SUPPORTED_BY target 须 Evidence)；`to_dict`/`from_dict` 快照。

**`graph_initializer.py`** — 新 operation 的初始 KG/AG
- `GraphInitializer.initialize` — 种子 `_build_initial_kg`(Host + NetworkZone + Goal + BELONGS_TO_ZONE/TARGETS)；`normalize_initial_target`/`initialize_graph_memory`。

**`graph_memory_store.py`** — 文件级图快照
- `GraphMemoryStore` — `load/save_kg`/`_ag`/`_runtime`/`save_snapshot`（每 op 一目录，json 存取）。

### `src/core/models/` —— 数据模型

**`kg.py`** — KG 节点/边模型
- `BaseGraphEntity`(id/label/properties/status/confidence/evidence_refs/first_seen/last_seen + `to_ref`/`validate_time_window`) → `BaseNode`/`BaseEdge`。
- 9 节点类：`Host`/`Service`/`Session`/`Evidence`/`Finding`/`NetworkZone`/`Goal`/`GoalProof`/`PivotRouteNode`。
- 4 边类：`HostsEdge`/`BelongsToZoneEdge`/`SupportedByEdge`/`TargetsEdge`。
- `NODE_MODEL_BY_TYPE`/`EDGE_MODEL_BY_TYPE` 映射 + `parse_node`/`parse_edge`；`GraphChange`/`GraphDelta` 变更记录。

**`kg_enums.py`** — `NodeType`(9)/`EdgeType`(4)/`EntityStatus`/`ChangeOperation`。
**`kg_exceptions.py`** — `KnowledgeGraphError` 及 Duplicate/NotFound/ValidationConstraint 子类。
**`kg_query.py`** / **`kg_types.py`** — 查询过滤模型 / 类型别名（Properties/JsonDict）。

**`ag.py`** — Attack Graph 容器
- `AttackGraph` — `add/get/list_nodes`/`_edges`/`find_process_nodes`/`to_dict`/`from_dict`(容错跳过未知类型)/`set_projection_metadata`；`parse_ag_node`/`parse_ag_edge`。

**`attack_process.py`** — AG 节点/边模型
- `AttackProcessNodeType`(ATTACK_STEP)/`AttackProcessEdgeType`(NEXT)；`AttackProcessNode` 基类 + `AttackStepNode`(每轮结果)；`AttackProcessEdge`。

**`runtime.py`** — RuntimeState 及其组成
- `RuntimeState`(sessions/credentials/pivot_routes/budgets/recent_outcomes/replan_requests；`drop_legacy_event_sourcing_fields` 容旧快照/`record_outcome`/`request_replan`)；`OperationRuntime`；`SessionRuntime`/`CredentialRuntime`/`PivotRouteRuntime`/`BudgetRuntime`/`OutcomeCacheEntry`/`ReplanRequest` + 各状态枚举。

**`graph_common.py`** — 共享图原语：`GraphRef`(graph/ref_id/ref_type + `key()`，**规范版**)、`stable_node_id`、`utc_now`。
**`finding.py`** — `Finding`/`RiskScore`/`EvidenceArtifactRecord`（报告层 finding 领域模型）。
**`scope.py`** — `Workspace`/`Engagement`/`Asset`/`ScopeRule`/`RiskPolicy` 等授权/范围模型。

### `src/core/evaluation/` —— 成功判定（声明式）

**`success_condition_tracker.py`** — `SuccessConditionTracker.evaluate`：对活的 KG/AG/Runtime 跑合约谓词，算 `_level_results`/`_achieved_level`/`_run_oracle`/`_check_chain_integrity`，产 `SuccessConditionProgress`。
**`predicate_engine.py`** — 谓词引擎
- `PredicateEngine.evaluate` + `@_register` 一组谓词：`exists_node`(+`_match_filters`/`_match_zone_ref` 按 kind/zone 判)、`count_nodes_at_least`、`exists_edge`/`path_exists`、`node_has_evidence`、`service_discovered_via_route`、`route_authorized`、`oracle_proof_valid`、`chain_satisfied`。
- `PredicateContext` — 提供 `nodes_from`(kg/ag)/`_runtime_nodes`(runtime→合成节点)/`resolve_zone_cidrs`/`pivot_routes_for_zone`。
**`goal_oracle.py`** — `GoalOracle.validate`：HMAC 校验 goal proof，只回 proof_token + redacted_summary（秘密不进图）。
**`models.py`** — `SuccessContract`/`ConditionBinding`/`SuccessConditionProgress`/`ConditionResult`/`GoalOracleInput/Output`/`OperationProfile`/`ZoneBinding`。
**`profile_loader.py`** / **`success_contract_loader.py`** — 从 YAML 载入 OperationProfile / SuccessContract。

### `src/core/llm/` —— LLM 客户端

**`packy_llm.py`** — 唯一 LLM 传输（OpenAI 兼容）
- `PackyLLMClient` — `complete_chat`/`chat`（httpx 调 base_url，`_post_with_retry` 重试、`_estimate_cost_usd` 计费、usage ledger）。
- `PackyLLMConfig`(api_key/base_url/model + `from_env`)、`PackyLLMResponse`、`ToolSpec`/`ToolCall`/`Message`、`PackyLLMError`、`load_llm_env_file`、usage ledger 一组函数。

### `src/core/validation/` —— ⚠️ 死代码（罐头验证时代残留，已核实**零活引用**）
`validation_plan.py`/`validation_result.py`/`vulnerability_profile.py`/`evidence_normalizer.py` — 安全验证 profile/plan/result 模型 + evidence 规整。mcp_lab 删除后全仓无 importer（src/tests 均 0）→ **可整包删除**（下一个清理项）。

### 其它
- `src/core/actions/__init__.py` — 仅 docstring 的空包（AG 驱动自动化占位）。
- `src/integrations/__init__.py` — 仅 docstring（罐头 `mcp_lab` 已删，真工具在外部 MCP server）。

---

## 五、配置与环境
- `configs/runtime_policy.full-chain.json` — full_chain 运行策略（authorized_hosts + adapter_policy.metasploit/pivot）。
- `configs/{exploit,vuln,fingerprint}_profiles/`、`configs/goal_oracles/` — 每 CVE/目标的 profile 与 oracle 配置。
- `lab/environments/full_chain_lab/` — 多主机靶场（docker-compose + success_contract.yml + profile.yml + msfrpcd sidecar override）。
- `cg.md` — 重构设计文档与迁移记录（附录 A–G，含 v3 六步迁移全过程）。

---

## 六、当前状态（六步 §8 全部到位）
Step1 去 lab 化 → Step2 收信封（channel-② 全删）→ Step3 planner agent 化（push）→ Step4 executor 放开（能力无关）→ Step5 纯真工具（塌缩节点到 `Evidence{kind}`）→ **Step6 LangGraph（三层活路径）**。
数据层已删到终态：KG 9 节点/4 边、AG 1 节点/1 边、无事件溯源、无 capability、无罐头工具/节点。
**剩余（与框架无关）**：工具硬化（装 nuclei、真 post_access/credential 工具、pivot register hint）+ full_chain E2E 拉绿。
