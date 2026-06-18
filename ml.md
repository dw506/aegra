# Aegra 代码结构与框架流程分析

> 生成日期：2026-06-17 · 对应当前 working tree（PEV v3 重构进行中）
> 关联设计文档：`docs/aegra_agentic_planner_graph_refactor.md`（v3 权威设计）

---

## 0. 项目是什么

Aegra 是一个**自动化授权渗透测试框架**。给定一个授权目标 + 自然语言任务目标，它用 LLM 驱动一个
**Plan → Execute → Verify（PEV）单层循环**自主推进：planner 决策、单执行器真打、确定性契约判成功。
执行器是**真实攻击者**（授权 scope 内默认放开利用/命令/会话/提权/横向/读数据），不是安全验证器。
两条硬不变量贯穿全局：**授权 scope 边界**、**密钥不落地（secrets 永不进图/日志/LLM）**。

整体被组织成**三层图**：
- **KG（Knowledge Graph）** = 世界状态真相（主机/服务/会话/凭证/漏洞/证据）。
- **AG（Attack Graph）** = 结果叙事时间线（每轮 1 个 `ATTACK_STEP` + 终态 `GOAL_OUTCOME`）。
- **Log（round log 文件）** = 执行过程明细（AG 只存 `log_ref` 指针）。

---

## 1. 顶层目录结构

```
Aegra/
├── src/                     # 全部源码（134 个 .py）
│   ├── app/                 # 应用层：编排入口 + HTTP API + 配置
│   ├── core/                # 领域核心（图/规划/执行/评估/运行时/模型…）
│   └── integrations/        # 外部集成（目前：mcp_lab 实验靶场工具服务）
├── tests/                   # 60 个测试文件（pytest）
├── configs/                 # 运行策略 / exploit profile / 成功契约样例
├── lab/                     # 靶场环境（full_chain_lab、Dockerfile、success_contract）
├── docs/                    # 设计文档（agentic_planner_graph_refactor 等）
├── runs/                    # e2e 测试夹具 run 日志（临时跑日志已 .gitignore）
├── scripts/                 # 运维 / 跑批脚本
├── web/                     # 可视化 dashboard 前端
├── var/ , tmp-runtime/      # 运行时产物（图快照、round log、artifacts）
├── Dockerfile, docker-compose.yml
├── pytest.ini, requirements.txt, openapi.json
└── ml.md（本文）, memory.md
```

---

## 2. 当前架构总览（运行时一张图）

```
                       ┌─────────────────────────────────────────────┐
   常驻极小摘要 ───────▶│  AppOrchestrator.run_operation_cycle (单循环) │
   (节点计数/eligible_   └─────────────────────────────────────────────┘
    for_stop/achieved_         │
    level/missing/最近3 step)  │  ① 读 KG/AG/Runtime + SuccessConditionTracker.evaluate
                               │  ② PlannerGraphTools.build_min_summary（极小摘要）
                               ▼
        ┌────────────────────────────┐  RoundDirective   ┌─────────────────────────┐
        │  MissionPlannerAgent.decide │ ────────────────▶ │  ExecutionAgent.run     │
        │  (唯一 LLM 决策点)          │                   │  (单执行器/单目标一轮)   │
        │  读: kg_query/ag_timeline.. │ ◀──────────────── │  ExecutionStageAgent    │
        │  写: record_finding /        │   RoundResult     │  内部有界多工具 ReAct    │
        │      record_attack_step /    │                   └─────────────────────────┘
        │      link_evidence (typed)   │                          │ 工具调用
        └────────────────────────────┘                          ▼
            │ stop/replan/execute                    ┌───────────────────────┐
            ▼                                         │  ToolGateway          │ ← 单一执行边界
   ┌──────────────────────────────────┐              │  无 pivot → MCP 透传   │
   │  PhaseTwoResultApplier            │              │  有 direct 路由 → A 适配器│
   │  apply_planner_decision (runtime) │              └───────────────────────┘
   │  apply_stage_result:              │                          │
   │   · KG: _fact_state_deltas        │              ┌───────────────────────┐
   │     → ToolTraceFactExtractor      │              │ ConfiguredMCPClient    │
   │   · AG: 单 ATTACK_STEP + NEXT 边  │              │ → mcp_lab 工具(真发包)  │
   │   · Runtime: 会话/路由/凭证       │              └───────────────────────┘
   └──────────────────────────────────┘
            │                  │                 │
            ▼                  ▼                 ▼
        ┌───────┐         ┌───────┐      ┌──────────────────┐
        │  KG   │◀── ADVANCED ─│ AG │     │ SuccessTracker   │
        │状态真相│         │结果叙事│────▶│ + PredicateEngine│
        └───────┘         └───────┘ log_ref│ + GoalOracle(HMAC)│
                              │             └──────────────────┘
                              ▼ round-N.txt（过程明细，AG 只存指针）
```

**一句话流程**：planner（带图读写工具）→ 一个 ExecutionAgent 跑完一个 capability 轮（经 ToolGateway 调真实工具）
→ 确定性抽事实写 KG / 单 step 写 AG / round log 落文件 → SuccessConditionTracker 刷新成功门 → planner 据判决决定 stop/继续。

---

## 3. 模块详解（按包）

### 3.1 `src/app/` —— 应用层（编排 + API + 配置）

| 模块 | 职责 |
|---|---|
| `orchestrator.py` | **核心**。`AppOrchestrator`：operation 生命周期（create/import_targets/start/run）+ 主循环 `run_operation_cycle`（第 4 节详述）+ 多轮 `run_until_quiescent` + 成功汇总 `get_operation_run_summary`。装配 planner、ExecutionAgent、ToolGateway、result_applier、图存储。 |
| `settings.py` | `AppSettings`：从环境/文件加载配置（MCP、LLM、runtime policy、lab profile、store 后端、审计/日志治理参数）。 |
| `llm_decision_observer.py` | LLM 决策观测/记录（审计 LLM 决策历史）。 |
| `api/operation_routes.py` | operation 生命周期 HTTP 路由（创建/导入目标/启动/运行/查询/findings/evidence/audit）。 |
| `api/execution_routes.py` | 执行与审批路由边界。 |
| `api/graph_routes.py` | 图与只读视图路由（KG/AG 快照、可视化）。 |

### 3.2 `src/core/planning/` —— 规划（thick agentic planner）

| 模块 | 职责 |
|---|---|
| `mission_planner_agent.py` | `MissionPlannerAgent.decide()`：唯一 LLM 决策点。输出 `PlannerOutcome`（action ∈ execute/stop_success/stop_failed/pause_for_review + `RoundDirective`）。在 decide 内通过 `graph_tools.apply_tool_calls(planner_tool_calls)` 调用写工具。 |
| `graph_tools.py` | **P3 typed 图工具** `PlannerGraphTools`：读（`kg_query`/`kg_get_node`/`kg_neighbors`/`ag_get_timeline`/`ag_get_step`/`get_round_log`）+ 写（`record_finding`/`link_evidence` 经 `kg.apply_patch_batch` 真写 KG；`record_attack_step` 记录语义意图）+ `build_min_summary`（常驻极小摘要）。LLM 只填叶子语义，ID/连边/脱敏/持久化由工具做。 |
| `llm_mission_planner_advisor.py` | `LLMMissionPlannerAdvisor`：把 planner 上下文 + 图工具摘要喂给 LLM，拿回决策 JSON。prompt 指示使用 graph_tools。 |
| `models.py` | `PlannerDecision` / `PlannerOutcome` / `RoundDirective` 互转模型。 |
| `prompts/planner_global_control.md` | planner 系统 prompt：驱动图工具、输出 directive、读 `eligible_for_stop`/`achieved_level` 自判 stop。 |

### 3.3 `src/core/stage/` + `src/core/execution/` —— 执行（单执行器 + 单一工具边界）

| 模块 | 职责 |
|---|---|
| `stage/agents/__init__.py` | **`ExecutionStageAgent`**：P2 合并后的**唯一**执行器（`accepts_any_request=True`，服务所有 capability 轮）。共享 `EXECUTION_AGENT_ROLE_PROMPT`（真实攻击姿态）。 |
| `stage/llm_driven_stage_agent.py` | `LLMDrivenStageAgent`：有界多工具 ReAct 循环——调 LLM 决定下一个工具 → 经 ToolGateway 调用 → 分析输出 → 直到 finish/max_steps，产出 `StageResult`（含 tool_trace/evidence/findings/会话/路由）。`accepts_any_request` 控制是否跳过 stage 绑定校验。 |
| `stage/registry.py` | `StageAgentRegistry`：现在持有**单个** ExecutionStageAgent；`resolve`/`resolve_agent` 对任意 stage/name 返回它；`validate_assignment` 单 agent 时 no-op。 |
| `stage/models.py` | `StageName`/`RoundDirective`/`RoundResult`/`StageExecutionRequest`/`StageResult`/`ToolTrace`/`ExtractedFact`/`CapabilityName`。 |
| `stage/adapters.py` | `StageResultAdapter`：`StageResult` → 规范 `AgentTaskResult`（供 ResultApplier 消费）。 |
| `execution/execution_agent.py` | `ExecutionAgent`：P2 门面。`RoundDirective` → `StageExecutionRequest`（capability→stage 映射 + `CAPABILITY_TOOLS` 工具表）→ 调单执行器 → 包成 `RoundResult` + 写 round-N.txt log。 |
| `execution/tool_gateway.py` | **`ToolGateway`**：单一执行边界，drop-in `MCPClient`。无 pivot → 透传底层 MCP（与直连等价）；路由声明 direct 传输(netns/tunnel/proxy) → 转 `ToolPlan` 走 A 适配器，结果适配回 `MCPToolCallResult`。 |
| `execution/configured_mcp_client.py` | `ConfiguredMCPClient`：JSON-RPC MCP 客户端（stdio/http），管理会话、tools 缓存、真实调用 mcp_lab 工具。 |
| `execution/mcp_client.py` | `MCPClient` 协议 + `MCPToolCallResult` + `UnavailableMCPClient`。 |
| `execution/pivot_context.py` | `PivotExecutionContextResolver`：把路由解析成代理 env / netns wrapper / 隧道端点（真实中转传输的基础件）。 |
| `execution/adapters/*` | A 的传输适配器：`netns_shell` / `tunnel` / `proxy_shell` / `local_shell` / `http_request` / `mcp`（`base.py` 定义 `ExecutionAdapter` 协议）。经 ToolGateway 在直接传输时调用。 |
| `execution/adapter_resolver.py` | `ToolAdapterResolver`：MCP-first 的 adapter 选择策略。 |
| `execution/tool_plan.py` / `tool_result.py` / `tool_policy.py` | `ToolPlan` / `ToolExecutionResult` / `ToolPolicy`：adapter 执行的输入/输出/策略模型。 |

### 3.4 `src/core/runtime/` —— 运行时（写回 / 管理器 / 审计）

| 模块 | 职责 |
|---|---|
| `result_applier.py` | **核心写回** `PhaseTwoResultApplier`。`apply_planner_decision`（只写 runtime meta + 审计，不写 AG）；`apply_stage_result`（① KG：`_fact_state_deltas`→唯一机器事实源；② AG：单 `ATTACK_STEP` + `NEXT` 边；③ Runtime：会话/路由/凭证/证据/findings）。写图与控制流解耦、逐条容错。 |
| `tool_trace_fact_extractor.py` | `ToolTraceFactExtractor`：从成功 ToolTrace 确定性抽取 KG 实体/关系（**唯一机器事实源**）。 |
| `txt_trace_logger.py` | `TxtTraceLogger`：append-only 分类文本轨迹（operation trace + 每轮 round log）。 |
| `store.py` | `RuntimeStore`（`InMemory`/`File`）：runtime state 持久化抽象。 |
| `reducer.py` | 把 runtime 事件 reduce 到 RuntimeState。 |
| `session_manager.py` / `lease_manager.py` / `pivot_route_manager.py` / `credential_manager.py` / `locks.py` / `reachability.py` | 运行时资源管理：会话/租约/中转路由/凭证/锁/可达性传播。 |
| `budgets.py` / `risk_scoring.py` / `approvals.py` | 预算守卫 / 风险打分 / 审批。 |
| `checkpoint_store.py` / `observability.py` / `llm_history.py` / `audit_report.py` / `report_generator.py` | 检查点/可恢复 / 可观测 / LLM 决策历史 / 审计报告 / findings 报告。 |
| `policy_engine.py` / `policy.py` | **`PolicyEngine`**：授权 scope 硬闸（`evaluate_target_scope`，唯一不可绕过的安全闸）+ task/tool/validator 风险闸（`RiskPolicy` 默认全放行，真实渗透姿态）。 |

### 3.5 `src/core/evaluation/` —— 成功判定（声明式契约 + 确定性评估）

| 模块 | 职责 |
|---|---|
| `success_condition_tracker.py` | `SuccessConditionTracker.evaluate()`：对照 agent 自建的 KG/AG/Runtime 求值契约 → `SuccessConditionProgress`（`eligible_for_stop`/`achieved_level`/`missing`/`satisfied`）。支持 `levels` 分级 + `require_chain` 不跳关。 |
| `predicate_engine.py` | `PredicateEngine`：抽象可验证谓词（`exists_node`/`count_nodes_at_least`(多主机量化)/`oracle_proof_valid`…）。 |
| `goal_oracle.py` | `GoalOracle`：用 env secret 对 agent 提交物做 HMAC 比对，只回 pass/fail + 不透明 proof_token（**密钥不落地**）。 |
| `models.py` | `SuccessContract`（`require_all`/`require_chain`/`levels`/`condition_bindings`）/ `SuccessConditionProgress`（`achieved_level`）。 |
| `success_contract_loader.py` / `profile_loader.py` | 从 YAML 加载契约 / OperationProfile。 |

### 3.6 `src/core/graph/` + `src/core/models/` —— 三层图与领域模型

| 模块 | 职责 |
|---|---|
| `graph/kg_store.py` | `KnowledgeGraph`：内存 KG（节点/边、`apply_patch_batch` 逐条容错、版本化）。 |
| `graph/graph_memory_store.py` | `GraphMemoryStore`：KG/AG/Runtime 快照文件持久化。 |
| `graph/graph_initializer.py` | 新 operation 的初始 KG/AG 构建（从导入目标）。 |
| `graph/topology.py` | 网络拓扑/区域辅助。 |
| `models/kg.py` / `kg_enums.py` / `kg_query.py` / `kg_types.py` | KG 节点/边模型、枚举（`NodeType`/`EdgeType`）、查询。 |
| `models/ag.py` / `attack_process.py` | AG 容器 + 结果层节点（`AttackStepNode`/`GoalOutcomeNode`，边 `NEXT`/`ADVANCED`）。 |
| `models/runtime.py` | `RuntimeState`/`OperationRuntime`/`SessionRuntime`/`PivotRouteRuntime`/`Budgets`…。 |
| `models/scope.py` | `Engagement`/`Asset`/`RiskPolicy`/`DenylistRule`（授权边界 + 风险策略默认值）。 |
| `models/finding.py` / `vulnerability_candidate.py` / `fingerprint.py` / `events.py` / `task_types.py` | findings / 漏洞候选 / 指纹 / 事件 / 任务类型模型。 |

### 3.7 感知 / 工具 / 漏洞候选 / 校验 / 反馈

| 模块 | 职责 |
|---|---|
| `perception/parser_registry.py` 等 | 工具结果解析插件（`tool_execution_parser` + `generic_parser`），把原始工具输出归一化。 |
| `tools/recipe.py` / `tool_runner.py` / `runner.py` / `registry.py` | 把 planner 任务提示转成安全工具配方并执行。 |
| `vuln_candidates/matcher.py` / `rules.py` | 指纹 → 漏洞候选匹配（Struts/Tomcat/Redis/Spring 等内置规则）。 |
| `validation/*` | 漏洞 profile / 验证计划 / 证据归一 / 验证结果模型。 |
| `feedback/result_verifier.py` / `evidence_extractor.py` | 受控工具结果的首轮校验 + 证据抽取。 |
| `capabilities/model.py` | stage 级能力模型。 |

### 3.8 `src/core/agents/` —— LLM 客户端与写回辅助

| 模块 | 职责 |
|---|---|
| `packy_llm.py` | `PackyLLMClient`：OpenAI 兼容 LLM 客户端。 |
| `state_writer.py` | `StateWriterAgent`：把感知记录归一成 KG delta + **KG store 写入器**（live 路径靠它把 fact deltas 落 store）。 |
| `agent_protocol.py` / `agent_models.py` / `kg_events.py` / `pipeline_results.py` | agent I/O 协议、记录模型、KG 事件、流水线结果壳。 |
| `llm_decision.py` / `llm_safety.py` | LLM 决策模型 / 安全辅助。 |

### 3.9 可视化 / 集成

| 模块 | 职责 |
|---|---|
| `visualization/unified_visualization.py` | 自动化运行可视化控制台只读适配器。 |
| `visualization/graph_publisher.py` / `graph_serializer.py` / `graph_event.py` | 图增量发布（WebSocket）/ 序列化 / 事件。 |
| `integrations/mcp_lab/server.py` / `tools.py` / `catalog.py` | 隔离靶场的 MCP 工具服务（newline-JSON-RPC）。`tools.py` 是**真发包**的 lab 工具（nmap/http/exploit/pivot/controlled_data_read…，含 GoalOracle HMAC 取证）。 |

---

## 4. 当前框架运行流程（端到端）

### 4.1 启动准备
1. `POST /operations` → `create_operation`：固化 control_plane / runtime_policy / lab_profile 到 metadata。
2. `import_targets`：用户目标 → `Asset` + `Engagement` scope_rules（授权边界）。
3. `start_operation`：`GraphInitializer` 从目标建初始 KG/AG（Host/Goal/Scope 节点）。

### 4.2 单轮主循环 `run_operation_cycle`（核心）
```
① 加载 KG / AG / RuntimeState（必要时 recover 脏状态）
② SuccessConditionTracker.evaluate(profile, contract, kg, ag, runtime)
        → success_condition_progress（eligible_for_stop / achieved_level / missing）
③ PlannerGraphTools.build_min_summary()  → 极小常驻摘要
④ PLAN：MissionPlannerAgent.decide(goal, summary, graph_tools)   ← 唯一 LLM 决策点
        · planner 可按需调读工具下钻 KG/AG
        · planner 在 decide 内调写工具（record_finding/link_evidence 真写 KG；record_attack_step 记意图）
        · 产出 PlannerOutcome：action + RoundDirective
⑤ apply_planner_decision → 写 runtime meta + 审计（不写 AG 节点）
⑥ 若 action == execute：
     ExecutionAgent.run(directive, pivot_routes, sessions, …)
        → ExecutionStageAgent 有界多工具 ReAct：
              调 LLM 选工具 → ToolGateway.call_tool（无 pivot 透传 MCP / direct 路由走适配器）
              → mcp_lab 工具真发包 → 分析输出 → 直到 finish/max_tools
        → 产出 StageResult + 写 round-N.txt（过程明细）
⑦ VERIFY + WRITE：apply_stage_result
        · KG：_fact_state_deltas → ToolTraceFactExtractor（唯一机器事实源）→ apply_patch_batch
        · AG：单 ATTACK_STEP 节点（capability/status/summary/kg_node_refs/log_ref）+ NEXT 边
        · Runtime：会话/中转路由/凭证/证据/findings
⑧ 再次 SuccessConditionTracker.evaluate 刷新成功门
⑨ 据 action 收尾：stop_success→COMPLETED / stop_failed→FAILED / pause→PAUSED / 否则 READY
⑩ 持久化 KG/AG/Runtime 快照 + 发布可视化增量
```

### 4.3 多轮 `run_until_quiescent`
循环调 `run_operation_cycle`，直到 planner 给 stop/pause、达 `max_cycles`/`max_replans`、或预算守卫触发。

### 4.4 成功如何判定（黑盒安全）
- 成功**声明式定义在契约**里（抽象可验证谓词，描述"成功的形状"而非答案）。
- `SuccessConditionTracker` + `PredicateEngine` 对照 **agent 自己建的 KG** 确定性求值。
- flag/token 只在 `GoalOracle` 内做 HMAC 比对，**永不进图/日志/LLM**。
- planner 只**读判决**（`eligible_for_stop`/`achieved_level`）决定何时 stop，**不自己判赢**。

---

## 5. 关键不变量（五条）
1. **授权 scope 边界**：`PolicyEngine.evaluate_target_scope` 硬闸，越界即拒（唯一不可绕过）。
2. **密钥不落地**：secrets 只在 GoalOracle 内 HMAC，永不进 KG/AG/Log/审计/LLM。
3. **成功权威在契约**：确定性追踪器算 `eligible_for_stop`/`achieved_level`，planner 只读不改。
4. **写图非控制闸门**：任何写图失败都不阻断下一步（`apply_patch_batch` 逐条容错）。
5. **黑盒完整性**：契约只暴露"成功的形状"与"目标类别"，不暴露答案/拓扑/技术路线。

---

## 6. 当前重构状态（2026-06-17）
- **P1 图瘦身 + 写图收敛**：✅ AG 砍 2 节点 2 边；KG 单一机器事实源；死的 `_structured_stage_state_deltas` 旁路已删。
- **P2 单执行器**：✅ 5 个 stage agent 合并成 1 个 `ExecutionStageAgent`；registry 持单 agent。
- **P3 planner 写工具**：✅ 已接入（record_finding/link_evidence 真写 KG，apply_tool_calls 在 decide 内调用）。
- **P4 契约分级**：✅ levels/achieved_level/require_chain/count_nodes_at_least 已实现。
- **单一执行边界 ToolGateway**：✅ B 接口 + A 传输合一（无 pivot 透传 MCP，direct 路由走适配器）。

**已知遗留 / 未接入**：
- gap 路由子系统（`gap_report`/`gap_router`/`plan_models`/`projection`）+ 旧 `src/core/prompts/` + 其他靶场环境（full_pentest/docker_lab/open_lab）+ runs/var 日志输出：**已于 2026-06-17 清理删除**，仅保留 `full_chain_lab` 测试环境。
- `result_applier.apply()` + `_run_state_writer`（obs→KG）：**仅测试用**，不在 live 路径。
- 测试：12 个 pre-existing 失败（`test_mcp_lab.py`×8 + `test_pivot_planning_context.py`×4），lab/contract 那块仍在迭代，与上述主链路改动无关。

---

## 7. 详细目录结构（文件级）

> 标注：`★`=主链路核心 · `▲`=本次重构改动 

```
src/
├── app/                                # 应用层
│   ├── orchestrator.py                 ★ AppOrchestrator：operation 生命周期 + 主循环 run_operation_cycle ▲
│   ├── settings.py                       AppSettings：环境/文件配置加载
│   ├── llm_decision_observer.py          LLM 决策观测/记录
│   └── api/
│       ├── __init__.py
│       ├── operation_routes.py           operation 生命周期 HTTP 路由
│       ├── execution_routes.py           执行/审批路由边界
│       └── graph_routes.py               图/只读视图路由
│
├── core/
│   ├── actions/__init__.py               动作占位包
│   │
│   ├── agents/                         # LLM 客户端 + 写回辅助
│   │   ├── packy_llm.py                   PackyLLMClient（OpenAI 兼容）
│   │   ├── state_writer.py             ★ StateWriterAgent：fact deltas → KG store 写入器
│   │   ├── agent_protocol.py             agent I/O 协议（AgentInput/Output/Context）
│   │   ├── agent_models.py               观测/证据/state-delta 记录模型
│   │   ├── kg_events.py                   KG delta 事件批
│   │   ├── pipeline_results.py            PipelineCycleResult/StepResult 结果壳
│   │   ├── llm_decision.py                LLM 决策模型（test-only 引用）
│   │   └── llm_safety.py                  LLM 安全辅助
│   │
│   ├── capabilities/
│   │   ├── __init__.py
│   │   └── model.py                       stage 级能力模型
│   │
│   ├── planning/                       # 规划（thick agentic planner）
│   │   ├── mission_planner_agent.py    ★ MissionPlannerAgent.decide：唯一 LLM 决策点
│   │   ├── graph_tools.py              ★▲ PlannerGraphTools：typed 读/写图工具（P3）
│   │   ├── llm_mission_planner_advisor.py LLM 决策 advisor
│   │   ├── models.py                     PlannerDecision/Outcome/RoundDirective
│   │   ├── __init__.py
│   │   └── prompts/planner_global_control.md  planner 系统 prompt
│   │
│   ├── stage/                          # 单执行器
│   │   ├── agents/__init__.py          ★▲ ExecutionStageAgent：合并后唯一执行器（P2）
│   │   ├── llm_driven_stage_agent.py   ★▲ 有界多工具 ReAct 循环
│   │   ├── registry.py                 ★▲ StageAgentRegistry：持单 agent
│   │   ├── models.py                     RoundDirective/RoundResult/StageExecutionRequest/StageResult/ToolTrace
│   │   ├── adapters.py                    StageResultAdapter → AgentTaskResult
│   │   └── __init__.py
│   │
│   ├── execution/                      # 单一工具边界 + 传输
│   │   ├── tool_gateway.py             ★▲ ToolGateway：唯一执行边界（B 接口+A 传输）
│   │   ├── execution_agent.py          ★▲ ExecutionAgent：directive→request 门面 + round log
│   │   ├── configured_mcp_client.py    ★ ConfiguredMCPClient：JSON-RPC MCP 客户端
│   │   ├── mcp_client.py                 MCPClient 协议 + MCPToolCallResult
│   │   ├── pivot_context.py              PivotExecutionContextResolver：中转传输解析
│   │   ├── adapter_resolver.py           ToolAdapterResolver：MCP-first 选择
│   │   ├── tool_plan.py / tool_result.py / tool_policy.py  adapter 输入/输出/策略
│   │   └── adapters/
│   │       ├── base.py                    ExecutionAdapter 协议
│   │       ├── netns_shell_adapter.py     netns 中转执行
│   │       ├── tunnel_adapter.py          隧道端点执行
│   │       ├── proxy_shell_adapter.py     代理(SOCKS/HTTP)执行
│   │       ├── local_shell_adapter.py     本地 shell
│   │       ├── http_request_adapter.py    HTTP 请求
│   │       └── mcp_adapter.py             MCP 适配器（Option 1 下冗余，暂留）
│   │
│   ├── runtime/                        # 写回 / 管理器 / 审计 / 策略
│   │   ├── result_applier.py           ★▲ PhaseTwoResultApplier：KG/AG/Runtime 写回
│   │   ├── tool_trace_fact_extractor.py ★ ToolTraceFactExtractor：唯一机器事实源
│   │   ├── txt_trace_logger.py         ★ TxtTraceLogger：operation/round log
│   │   ├── policy_engine.py            ★ PolicyEngine：scope 硬闸 + 风险闸
│   │   ├── policy.py                     RuntimePolicy/PolicyDecision
│   │   ├── store.py                    ★ RuntimeStore（InMemory/File）
│   │   ├── reducer.py                    runtime 事件 reduce
│   │   ├── session_manager.py            会话管理
│   │   ├── lease_manager.py              租约管理
│   │   ├── pivot_route_manager.py        中转路由管理
│   │   ├── credential_manager.py         凭证管理
│   │   ├── locks.py / reachability.py     锁 / 可达性传播
│   │   ├── budgets.py / risk_scoring.py / approvals.py  预算/风险/审批
│   │   ├── checkpoint_store.py / observability.py  检查点/可观测
│   │   ├── llm_history.py / audit_report.py / report_generator.py  LLM 历史/审计/报告
│   │   ├── events.py / runtime_queries.py / worker_result_adapter.py
│   │
│   ├── evaluation/                     # 成功判定（契约 + 确定性评估）
│   │   ├── success_condition_tracker.py ★ SuccessConditionTracker：契约求值 + 分级(P4)
│   │   ├── predicate_engine.py         ★ PredicateEngine：抽象可验证谓词
│   │   ├── goal_oracle.py              ★ GoalOracle：HMAC 取证（密钥不落地）
│   │   ├── models.py                     SuccessContract/Progress（levels/achieved_level）
│   │   ├── success_contract_loader.py / profile_loader.py  YAML 加载
│   │   └── __init__.py
│   │
│   ├── graph/                          # 三层图存储
│   │   ├── kg_store.py                 ★ KnowledgeGraph：内存 KG + apply_patch_batch
│   │   ├── graph_memory_store.py       ★ GraphMemoryStore：KG/AG/Runtime 快照持久化
│   │   ├── graph_initializer.py          初始 KG/AG 构建
│   │   └── topology.py                   拓扑/区域辅助
│   │
│   ├── models/                         # 领域模型
│   │   ├── kg.py / kg_enums.py / kg_query.py / kg_types.py / kg_exceptions.py  KG 模型/枚举
│   │   ├── ag.py                       ★ AttackGraph 容器 + parse_ag_node/edge
│   │   ├── attack_process.py           ★ AttackStepNode/GoalOutcomeNode + NEXT/ADVANCED 边
│   │   ├── runtime.py                  ★ RuntimeState/Session/PivotRoute/Budgets
│   │   ├── scope.py                    ★ Engagement/Asset/RiskPolicy ▲(权限默认值修复)
│   │   ├── finding.py / vulnerability_candidate.py / fingerprint.py  findings/漏洞/指纹
│   │   ├── events.py / task_types.py / graph_common.py
│   │
│   ├── perception/                     # 工具结果解析
│   │   ├── parser_registry.py            解析插件注册
│   │   ├── tool_execution_parser.py      工具执行解析（优先）
│   │   ├── generic_parser.py / parser_protocol.py
│   │   └── __init__.py
│   │
│   ├── tools/                          # 安全工具配方/执行
│   │   ├── recipe.py                     planner 提示 → 安全工具配方
│   │   ├── tool_runner.py / runner.py / registry.py
│   │   └── __init__.py
│   │
│   ├── vuln_candidates/                # 漏洞候选匹配
│   │   ├── matcher.py                    指纹 → 漏洞候选
│   │   ├── rules.py                      内置规则（Struts/Tomcat/Redis/Spring…）
│   │   └── __init__.py
│   │
│   ├── validation/                     # 漏洞校验
│   │   ├── vulnerability_profile.py / validation_plan.py / validation_result.py / evidence_normalizer.py
│   │   └── __init__.py
│   │
│   ├── feedback/                       # 结果校验/证据抽取
│   │   ├── result_verifier.py / evidence_extractor.py
│   │   └── __init__.py
│   │
│   ├── visualization/                  # 可视化
│   │   ├── unified_visualization.py      运行可视化只读适配器
│   │   ├── graph_publisher.py / graph_serializer.py / graph_event.py  图增量发布/序列化
│   │   └── __init__.py
│   │
│
└── integrations/
    ├── __init__.py
    └── mcp_lab/                        # 隔离靶场 MCP 工具服务
        ├── server.py                     newline-JSON-RPC MCP server
        ├── tools.py                    ★ 真发包 lab 工具（nmap/http/exploit/pivot/controlled_data_read + GoalOracle HMAC）
        ├── catalog.py                    默认开放工具目录
        ├── http_server.py                HTTP 传输入口
        └── __init__.py
```

