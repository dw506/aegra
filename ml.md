# Aegra 代码目录与模块职责说明

本文用于快速理解当前代码结构、模块边界和主要文件职责。整体系统围绕 `KG -> AG -> TG -> Scheduler -> Worker -> Execution -> Perception -> ResultApplier -> Runtime audit` 的闭环组织。

## 顶层目录

```text
D:\Aegra
├── configs/        # 外部集成和运行配置示例
│   └── incalmo.yaml                    # Incalmo C2 集成配置示例
├── docs/           # 架构和运行流程文档
│   └── aegra_runtime_flow.md           # Runtime 主链路、边界和 guardrail 说明
├── src/            # 主源码
│   ├── app/                            # 应用层：配置、API、编排器、静态前端
│   │   ├── orchestrator.py             # AppOrchestrator，operation 生命周期和主 control cycle 编排
│   │   ├── settings.py                 # AppSettings，环境变量/运行配置/LLM/MCP 配置入口
│   │   ├── llm_decision_observer.py    # 从 pipeline 输出提取 LLM decision history
│   │   ├── api/                        # HTTP API 注册层
│   │   │   ├── __init__.py             # FastAPI 应用主入口和当前主要 API surface
│   │   │   ├── operation_routes.py     # operation lifecycle 路由边界占位
│   │   │   ├── execution_routes.py     # execution/approval 路由边界占位
│   │   │   └── graph_routes.py         # graph/read-model 路由边界占位
│   │   └── static/                     # 本地控制台前端静态文件
│   │       ├── index.html              # 控制台 HTML 入口
│   │       ├── app.js                  # 前端交互和 API 调用逻辑
│   │       └── styles.css              # 控制台样式
│   ├── core/                           # 核心领域逻辑
│   │   ├── actions/                    # AG action 到 TG task 的确定性模板
│   │   │   ├── __init__.py             # action template API 导出
│   │   │   └── schemas.py              # ActionTemplate/Input/Output 和任务构造逻辑
│   │   ├── agents/                     # Agent 协议、pipeline、planner、critic、LLM advisor
│   │   │   ├── agent_protocol.py       # AgentInput/Output、权限、上下文和 BaseAgent
│   │   │   ├── agent_models.py         # Agent 层共享 record 模型
│   │   │   ├── registry.py             # Agent 注册、查找和 dispatch
│   │   │   ├── agent_pipeline.py       # Planner/Builder/Scheduler/Worker/Feedback/Supervisor 串联
│   │   │   ├── pipeline_builders.py    # 可选 pipeline 装配和 LLM advisor 注入
│   │   │   ├── planner.py              # PlannerAgent，生成 planning candidates
│   │   │   ├── task_builder.py         # TaskBuilderAgent，planner output -> TG patch
│   │   │   ├── scheduler_agent.py      # SchedulerAgent，TG + Runtime -> worker assignment
│   │   │   ├── perception.py           # PerceptionAgent，worker result -> observation/evidence
│   │   │   ├── state_writer.py         # StateWriterAgent，observation/evidence -> KG delta
│   │   │   ├── graph_projection.py     # GraphProjectionAgent，KG delta -> AG delta
│   │   │   ├── kg_events.py            # StateWriter 与 GraphProjection 间的 KG event 模型
│   │   │   ├── critic.py               # CriticAgent，失败/重复/低价值路径分析和 replan 建议
│   │   │   ├── supervisor.py           # SupervisorAgent，高层控制策略建议
│   │   │   ├── graph_context.py        # 构造给 Graph LLM 的紧凑图上下文，包含拓扑/可达路径/pivot route 摘要
│   │   │   ├── graph_llm_models.py     # Graph LLM proposal/validation 模型
│   │   │   ├── graph_llm_planner.py    # Graph-driven LLM planner advisor
│   │   │   ├── llm_decision.py         # LLM decision 模型、validator 和 forbidden key 校验
│   │   │   ├── llm_safety.py           # LLM prompt/response 脱敏、长度和深度限制
│   │   │   ├── packy_llm.py            # Packy/OpenAI-compatible 底层 LLM client
│   │   │   ├── packy_planner_advisor.py    # Planner rank LLM advisor
│   │   │   ├── packy_critic_advisor.py     # Critic LLM advisor
│   │   │   └── packy_supervisor_advisor.py # Supervisor LLM advisor
│   │   ├── planner/                    # 确定性规划内核，不依赖 agents/LLM
│   │   │   ├── README.md               # planner kernel 边界规则
│   │   │   ├── planner.py              # AttackGraphPlanner，AG 路径搜索和 task candidate 导出
│   │   │   ├── scorer.py               # HeuristicScorer 和 scoring context
│   │   │   └── critic.py               # TaskGraphCritic，确定性 TG 批评和 replan frontier
│   │   ├── graph/                      # KG/AG/TG 构建、投影、合并和快照
│   │   │   ├── kg_store.py             # 内存 KnowledgeGraph 实现
│   │   │   ├── ag_projector.py         # KG -> AG 投影
│   │   │   ├── tg_builder.py           # AG -> TG task graph 生成
│   │   │   ├── tg_merge.py             # TG frontier 增量合并并保留生命周期
│   │   │   ├── graph_initializer.py    # 新 operation 初始 KG/AG/TG 构建
│   │   │   ├── topology.py             # KG 网络拓扑查询，提取 reachability path、route candidate、zone 归属
│   │   │   └── graph_memory_store.py   # 文件型 KG/AG/TG/Runtime 快照存储
│   │   ├── models/                     # 核心领域 Pydantic 模型
│   │   │   ├── kg.py                   # KG 节点/边模型，含 CAN_REACH/PIVOTS_TO 拓扑字段
│   │   │   ├── kg_enums.py             # KG 枚举
│   │   │   ├── kg_exceptions.py        # KG 异常
│   │   │   ├── kg_query.py             # KG 查询模型
│   │   │   ├── kg_types.py             # KG 类型别名
│   │   │   ├── ag.py                   # Attack Graph 模型和容器
│   │   │   ├── tg.py                   # Task Graph 模型、状态、依赖和生命周期
│   │   │   ├── runtime.py              # RuntimeState、OperationRuntime、TaskRuntime、PivotRouteRuntime 等
│   │   │   ├── events.py               # Worker/result/fact/projection/runtime request 模型
│   │   │   ├── finding.py              # Finding、Evidence、RiskScore 模型
│   │   │   ├── fingerprint.py          # Service/component fingerprint 模型
│   │   │   ├── scope.py                # Asset、Engagement、scope policy 模型
│   │   │   └── vulnerability_candidate.py # 漏洞候选模型
│   │   ├── runtime/                    # RuntimeState、调度、策略、审计、报告、恢复
│   │   │   ├── store.py                # RuntimeStore，内存/文件存储
│   │   │   ├── scheduler.py            # Runtime scheduler、policy guardrail 和 route-aware gate
│   │   │   ├── policy.py               # RuntimePolicy schema 和加载逻辑
│   │   │   ├── policy_engine.py        # scope/tool/safety 策略评估
│   │   │   ├── policy_gate.py          # 调度前轻量 policy gate
│   │   │   ├── budgets.py              # 成本/风险/噪声预算控制
│   │   │   ├── locks.py                # 资源锁管理
│   │   │   ├── lease_manager.py        # session lease 管理
│   │   │   ├── session_manager.py      # runtime session 生命周期管理
│   │   │   ├── credential_manager.py   # runtime credential 状态管理
│   │   │   ├── pivot_route_manager.py  # pivot route 管理，按 host/zone/cidr/端口/协议选择可用路径
│   │   │   ├── reachability.py         # worker reachability 输出 -> Runtime pivot route 状态同步
│   │   │   ├── events.py               # runtime event 模型
│   │   │   ├── reducer.py              # runtime event -> RuntimeState reducer
│   │   │   ├── result_applier.py       # worker result -> Runtime/KG/AG/TG/audit 的边界
│   │   │   ├── observability.py        # operation log/checkpoint/shutdown/脱敏
│   │   │   ├── checkpoint_store.py     # runtime checkpoint 管理
│   │   │   ├── approvals.py            # 人工审批请求模型
│   │   │   ├── runtime_queries.py      # 只读 runtime 查询服务
│   │   │   ├── repetition_detector.py  # 确定性重复任务检测
│   │   │   ├── risk_scoring.py         # Finding 风险评分
│   │   │   ├── report_generator.py     # 报告生成
│   │   │   ├── audit_report.py         # operation audit report 汇总
│   │   │   └── llm_history.py          # LLM decision history 存储模型和查询
│   │   ├── execution/                  # ToolPlan 执行、MCP-first resolver 和 adapter dispatch
│   │   │   ├── __init__.py             # execution API 导出
│   │   │   ├── adapter_resolver.py     # ToolAdapterResolver，按 policy/binding/MCP-first/fallback 选择 adapter
│   │   │   ├── mcp_client.py           # MCP client 协议、标准 MCP tool call result 和 unavailable 默认实现
│   │   │   ├── pivot_context.py        # pivot route/session -> adapter execution context
│   │   │   ├── tool_plan.py            # TG task -> adapter-neutral ToolPlan，adapter 可为空
│   │   │   ├── tool_policy.py          # ToolPlan 执行前策略检查，含 MCP allow/deny 检查
│   │   │   ├── tool_result.py          # adapter-neutral 执行结果模型
│   │   │   ├── executor.py             # ExecutionExecutor，通过 resolver 选择 adapter 并执行
│   │   │   └── adapters/               # 执行 adapter
│   │   │       ├── __init__.py         # adapter API 导出
│   │   │       ├── base.py             # ExecutionAdapter 协议
│   │   │       ├── mcp_adapter.py      # MCPExecutionAdapter，调用 MCP tool 并转换为 ToolExecutionResult
│   │   │       ├── local_shell_adapter.py  # 本地 shell/subprocess adapter
│   │   │       ├── incalmo_c2_adapter.py   # Incalmo C2 adapter
│   │   │       ├── proxy_shell_adapter.py  # SOCKS/HTTP proxy 环境注入 adapter
│   │   │       ├── tunnel_adapter.py       # 预开 TCP tunnel endpoint adapter
│   │   │       └── netns_shell_adapter.py  # Linux network namespace shell adapter
│   │   ├── perception/                 # raw result -> observation/evidence parser
│   │   │   ├── __init__.py             # perception API 导出
│   │   │   ├── parser_protocol.py      # parser 协议和结果模型
│   │   │   ├── parser_registry.py      # parser registry 和 fallback 选择
│   │   │   ├── generic_parser.py       # 通用 fallback parser
│   │   │   ├── tool_execution_parser.py # ToolExecutionResult parser
│   │   │   └── c2_parser.py            # Incalmo C2 parser 兼容 shim
│   │   ├── feedback/                   # evidence 提取和结果校验
│   │   │   ├── __init__.py             # 包标识
│   │   │   ├── evidence_extractor.py   # 从受控工具输出提取证据
│   │   │   └── result_verifier.py      # controlled result 初步验证
│   │   ├── tools/                      # 受控 tool recipe
│   │   │   ├── __init__.py             # 包标识
│   │   │   ├── recipe.py               # ToolTask/ToolRecipe 和安全转换
│   │   │   ├── registry.py             # recipe registry
│   │   │   └── runner.py               # recipe 执行器
│   │   ├── vuln_candidates/            # fingerprint -> 漏洞候选
│   │   │   ├── __init__.py             # matcher/rules 导出
│   │   │   ├── rules.py                # 内置候选规则
│   │   │   └── matcher.py              # CandidateMatcher
│   │   └── workers/                    # Worker agent、legacy wrapper、domain service
│   │       ├── README.md               # worker 边界说明
│   │       ├── base.py                 # BaseWorkerAgent、任务规格、结果封装
│   │       ├── registry.py             # primary worker registry
│   │       ├── access_validation_worker.py     # access validation primary worker
│   │       ├── pivot_validation_worker.py      # pivot route probe/health primary worker
│   │       ├── goal_validation_worker.py       # goal validation primary worker
│   │       ├── privilege_validation_worker.py  # privilege validation primary worker
│   │       ├── port_scan_worker.py             # 受控端口扫描 primary worker
│   │       ├── web_enum_worker.py              # 低风险 Web 枚举 primary worker
│   │       ├── web_discovery_worker.py         # 目录/API 发现 primary worker
│   │       ├── credential_validation_worker.py # 凭据有效性验证 primary worker
│   │       ├── credential_reuse_worker.py      # 凭据复用检测 primary worker
│   │       ├── lateral_reachability_worker.py  # 横向可达性验证 primary worker
│   │       ├── internal_service_fingerprint_worker.py # 内网服务 fingerprint primary worker
│   │       ├── fingerprint_worker.py           # fingerprint 归一化和漏洞候选生成
│   │       ├── recon_worker.py                 # host/service/context recon worker
│   │       ├── vulnerability_validation_worker.py # controlled vulnerability validation worker
│   │       ├── probe_adapters.py        # nmap/httpx/whatweb/custom probe adapter
│   │       ├── tool_runner.py           # subprocess tool helper
│   │       ├── access_validators.py     # access validator adapter
│   │       ├── goal_validator.py        # goal validator 抽象
│   │       ├── goal_command_validator.py # command-backed goal validator
│   │       ├── access_worker.py         # legacy access worker
│   │       ├── goal_worker.py           # legacy goal worker
│   │       ├── services/                # worker 业务逻辑单一来源
│   │       │   ├── __init__.py          # 包标识
│   │       │   ├── access_validation_service.py    # access validation 业务逻辑
│   │       │   ├── goal_validation_service.py      # goal validation 业务逻辑
│   │       │   ├── privilege_validation_service.py # privilege validation 业务逻辑
│   │       │   └── result_builders.py   # worker service 共享结果模型
│   │       └── vulnerability_validators/ # 漏洞验证器插件
│   │           ├── __init__.py          # validator API 导出
│   │           ├── base.py              # ValidationTarget/Result/Validator 抽象
│   │           ├── registry.py          # validator registry
│   │           ├── http_fingerprint.py  # 低风险 HTTP fingerprint validator
│   │           ├── default_credentials_idor.py # 默认凭据/IDOR 受控验证器
│   │           └── struts2_s2045.py     # Struts2 S2-045 validator
│   └── integrations/                    # 外部系统集成
│       ├── __init__.py                  # 集成包标识
│       └── incalmo/                     # Incalmo C2 协议集成
│           ├── __init__.py              # Incalmo API 导出
│           ├── models.py                # Incalmo protocol model
│           ├── client.py                # Incalmo HTTP client
│           ├── mapper.py                # Incalmo <-> Aegra result/runtime 映射
│           └── perception.py            # Incalmo C2 output parser 插件
├── tests/          # 单元、集成、烟测测试
│   ├── integration/                     # 需要外部服务/显式 env 开关的集成测试
│   │   └── test_incalmo_c2_live.py      # Incalmo C2 live smoke test
│   ├── test_app_orchestrator.py         # AppOrchestrator 和应用配置测试
│   ├── test_api.py                      # API 基础行为测试
│   ├── test_api_operation_cycle.py      # operation cycle API 测试
│   ├── test_approval_api.py             # approval API 测试
│   ├── test_findings_api.py             # findings/evidence API 测试
│   ├── test_product_api_ui.py           # 静态前端/API smoke 测试
│   ├── test_agents.py                   # Agent 通用行为和 advisor 集成测试
│   ├── test_planner.py                  # PlannerAgent 测试
│   ├── test_critic.py                   # CriticAgent 测试
│   ├── test_supervisor_agent.py         # SupervisorAgent 和控制策略测试
│   ├── test_pipeline_builders.py        # pipeline 装配测试
│   ├── test_packy_llm.py                # Packy LLM client 测试
│   ├── test_packy_planner_advisor.py    # Planner LLM advisor 测试
│   ├── test_packy_critic_advisor.py     # Critic LLM advisor 测试
│   ├── test_graph_llm_models.py         # Graph LLM 模型测试
│   ├── test_graph_llm_planner.py        # Graph LLM planner advisor 测试
│   ├── test_llm_decision_models.py      # LLM decision validator 测试
│   ├── test_llm_decision_history.py     # LLM decision history 测试
│   ├── test_ag_models.py                # AG 模型测试
│   ├── test_ag_projector.py             # KG -> AG 投影测试
│   ├── test_tg_models.py                # TG 模型测试
│   ├── test_tg_builder.py               # TG builder 测试
│   ├── test_tg_merge.py                 # TG merge 测试
│   ├── test_tg_lifecycle_progression.py # TG lifecycle progression 测试
│   ├── test_graph_initializer.py        # 初始图构建测试
│   ├── test_graph_memory_store.py       # Graph memory store 测试
│   ├── test_graph_context.py            # GraphContext builder 测试
│   ├── test_network_topology.py         # KG 网络拓扑查询测试
│   ├── test_runtime_state.py            # RuntimeState 模型测试
│   ├── test_runtime_store.py            # RuntimeStore 测试
│   ├── test_runtime_scheduler.py        # Runtime scheduler 测试
│   ├── test_runtime_budgets.py          # 预算控制测试
│   ├── test_runtime_locks.py            # resource lock 测试
│   ├── test_runtime_leases.py           # lease manager 测试
│   ├── test_runtime_sessions.py         # session manager 测试
│   ├── test_runtime_credentials.py      # credential manager 测试
│   ├── test_runtime_events.py           # runtime events/reducer 测试
│   ├── test_runtime_checkpoint_store.py # checkpoint store 测试
│   ├── test_runtime_pivot_routes.py     # pivot route manager 测试
│   ├── test_agent_workers.py            # worker agent 输出和协议测试
│   ├── test_phase_two_workers.py        # phase-two worker/service 测试
│   ├── test_phase_two_result_applier.py # ResultApplier 测试
│   ├── test_access_service_execution.py # access service 执行测试
│   ├── test_worker_registry.py          # WorkerRegistry 测试
│   ├── test_extended_workers.py         # 新增受控 worker 行为测试
│   ├── test_worker_expansion_tg_policy.py # 新 worker 的 TG 映射/资源锁/policy 测试
│   ├── test_tool_execution.py           # ToolPlan/executor/MCP adapter/resolver 测试
│   ├── test_pivot_execution_context.py  # pivot execution context/adapter/runtime request 测试
│   ├── test_tool_policy.py              # tool policy 测试
│   ├── test_tool_execution_parser.py    # ToolExecutionResult parser 测试
│   ├── test_tool_execution_runtime_audit.py # tool execution audit 测试
│   ├── test_tool_recipe_feedback.py     # tool recipe feedback 测试
│   ├── test_fingerprint_worker.py       # fingerprint worker 测试
│   ├── test_fingerprint_adapters.py     # probe/fingerprint adapter 测试
│   ├── test_vulnerability_candidate_matcher.py # 漏洞候选匹配测试
│   ├── test_vulnerability_validation_worker.py # 漏洞验证 worker 测试
│   ├── test_vulnerability_validator_models.py  # 漏洞验证模型测试
│   ├── test_vulnerability_validator_registry.py # validator registry 测试
│   ├── test_scope_policy.py             # scope policy 测试
│   ├── test_policy_gate.py              # policy gate 测试
│   ├── test_policy_scheduler_gate.py    # scheduler policy gate 测试
│   ├── test_report_generator.py         # 报告生成测试
│   ├── test_audit_report.py             # 审计报告测试
│   ├── test_risk_scoring.py             # 风险评分测试
│   ├── test_repetition_detector.py      # 重复任务检测测试
│   ├── test_action_templates.py         # action template 测试
│   ├── test_candidate_to_task_graph.py  # candidate -> TG 测试
│   ├── test_execution_end_to_end_smoke.py # execution/perception/runtime audit smoke
│   ├── test_incalmo_c2_adapter.py       # Incalmo adapter 测试
│   ├── test_incalmo_integration.py      # Incalmo integration 测试
│   ├── test_incalmo_settings.py         # Incalmo 与 MCP settings 测试
│   ├── test_architecture_boundaries.py  # 架构依赖边界测试
│   ├── test_perception_boundaries.py    # perception 边界测试
│   ├── test_nuclei_safe_adapter.py      # nuclei safe adapter 测试
│   ├── test_scorer.py                   # planner scorer 测试
│   ├── test_findings.py                 # Finding 模型/逻辑测试
│   ├── test_vulhub_kg_smoke.py          # Vulhub KG smoke 测试
│   ├── test_vulhub_orchestrator_smoke.py # Vulhub orchestrator smoke 测试
│   └── test_graph_iteration_vuln_env_smoke.py # 图迭代漏洞环境 smoke 测试
├── var/            # 本地运行态、审计、图快照等生成数据
├── requirements.txt
├── memory.md       # 项目结构/迁移记忆
└── ml.md           # 本文件
```

## 核心运行链路

```text
AppOrchestrator
  -> AgentPipeline
  -> PlannerAgent
  -> TaskBuilderAgent
  -> SchedulerAgent / RuntimeScheduler route-aware gate
  -> WorkerRegistry / BaseWorkerAgent
  -> Worker Service
  -> ExecutionExecutor / ExecutionAdapter
  -> ToolExecutionResult
  -> Perception / ParserRegistry
  -> PhaseTwoResultApplier / ReachabilityPropagator
  -> RuntimeState + KG/AG/TG 更新 + 审计
```

LLM 只通过 advisor 层进入系统：

- Planner rank advisor：对已有候选排序和解释增强。
- Graph LLM planner advisor：基于图上下文提出声明式 task proposal；图上下文包含 KG reachability path、network zone 和 Runtime pivot route 摘要。
- Critic advisor：总结失败原因和 replan hint。
- Supervisor advisor：在有限策略集合中给高层控制建议。

LLM 不应直接执行命令、生成 shell payload、写 KG/AG/TG/Runtime。

## `configs/`

- `incalmo.yaml`：Incalmo C2 集成配置示例。

## `docs/`

- `aegra_runtime_flow.md`：当前运行闭环、所有权边界和 guardrail 说明。

## `src/app/` 应用层

应用层负责把核心能力装配成可运行的本地控制面：配置、API、operation 生命周期、运行主循环、审计查询和前端静态资源。

- `settings.py`：环境变量和运行配置入口。包含 runtime store、审计、策略、LLM、MCP、工具路径、Incalmo 等配置。MCP 相关字段包括 `mcp_enabled`、`mcp_first`、`mcp_config_path`、`mcp_config_json`、`mcp_default_timeout_seconds` 和 `allow_local_fallback`，并支持 `AEGRA_MCP_ENABLED`、`AEGRA_MCP_FIRST`、`AEGRA_MCP_CONFIG_PATH`、`AEGRA_MCP_CONFIG_JSON`、`AEGRA_MCP_DEFAULT_TIMEOUT_SECONDS`、`AEGRA_ALLOW_LOCAL_FALLBACK`。
- `orchestrator.py`：应用级编排门面。负责 operation 创建/启动/停止、目标导入、运行 control cycle、加载/保存图记忆、调用 pipeline、应用 worker 结果、记录审计和控制摘要。
- `llm_decision_observer.py`：从 pipeline cycle 输出中提取 prompt-free 的 LLM decision history，负责记录 accepted/rejected、decision id、target id、模型和 usage 等审计摘要。

### `src/app/api/`

API 层负责暴露本地控制面 HTTP 接口。

- `__init__.py`：FastAPI 应用主入口和当前主要 API surface，注册 operation、approval、findings、audit、LLM history 等路由。
- `operation_routes.py`：operation lifecycle 路由边界占位，当前大部分仍由 `__init__.py` 兼容注册。
- `execution_routes.py`：execution/approval 路由边界占位。
- `graph_routes.py`：graph/read-model 路由边界占位。

### `src/app/static/`

本地控制台静态前端。

- `index.html`：控制台页面入口。
- `app.js`：前端交互逻辑，调用后端 operation/API。
- `styles.css`：控制台样式。

## `src/core/actions/` 动作模板

负责把 AG action 转换为 TG task 的确定性模板。

- `__init__.py`：导出 action template API。
- `schemas.py`：定义 `ActionTemplateInput/Output`、`ActionTemplate`，以及 `build_task_from_action` 等模板构造逻辑。

## `src/core/agents/` Agent 层

Agent 层是系统的决策和协议层。每个 Agent 通过 `AgentInput/AgentOutput` 通信，输出结构化 decisions、state_deltas、replan_requests、evidence、logs 等，不直接越权写其他图结构。

- `agent_protocol.py`：Agent 通用协议、上下文、权限模型、输入输出 envelope、基类。
- `agent_models.py`：Agent 层共享记录模型，如 decision、outcome、observation、state delta、replan request。
- `registry.py`：Agent 注册、查找和 dispatch。
- `agent_pipeline.py`：最小 pipeline 编排器，串联 planner、task builder、scheduler、worker、feedback、supervisor cycle。
- `pipeline_builders.py`：可选 pipeline 装配逻辑，根据配置启用 Packy/Graph LLM advisor、critic advisor、supervisor advisor。

### 规划与任务生成

- `planner.py`：PlannerAgent。读取 AG/goal/graph_context，生成 planning candidates；可选消费 LLM 排序建议和 Graph LLM task proposal，但不写 TG、不执行工具。
- `task_builder.py`：TaskBuilderAgent。把 planner 选中的 candidate/action chain 转成 TG patch delta。
- `scheduler_agent.py`：SchedulerAgent。读取 TG 和 RuntimeState，找出可调度任务并生成 worker assignment decisions；实际 route-aware admissibility 由 `RuntimeScheduler` 执行。

### 反馈、图投影与状态写入

- `perception.py`：PerceptionAgent。把 worker outcome 和 raw result 解释成 observation/evidence，不直接写图。
- `state_writer.py`：StateWriterAgent。把 observation/evidence 标准化为 KG delta payload，是 Agent 层正式 KG writer 边界。
- `graph_projection.py`：GraphProjectionAgent。把 KG delta event 映射为 AG delta，并产生后续投影事件。
- `kg_events.py`：StateWriter 和 GraphProjection 之间传递的 KG delta event 模型。

### Critic / Supervisor

- `critic.py`：CriticAgent。分析 TG/Runtime 执行质量，发现重复、失败饱和、低价值路径、失效证据等，输出建议和 replan request。
- `supervisor.py`：SupervisorAgent。高层控制策略建议，只能在有限策略中选择，如继续规划、继续执行、请求重规划、暂停复核等。

### LLM 支撑层

- `packy_llm.py`：Packy/OpenAI-compatible 底层 LLM 客户端，负责 chat completions 调用、SSE/JSON 文本解析、usage/cost 统计。
- `packy_planner_advisor.py`：Planner rank LLM advisor，把 LLM 输出转换为受限的候选排序/解释增强建议。
- `graph_llm_planner.py`：Graph-driven LLM planner advisor，基于 compact `GraphContext` 生成声明式 graph task proposal，并先校验再交给 Planner。
- `graph_llm_models.py`：Graph LLM proposal、task proposal、rank adjustment、validation result 等结构模型。
- `packy_critic_advisor.py`：Critic LLM advisor，把 LLM 输出转换为 finding summary、rationale suffix、replan hint。
- `packy_supervisor_advisor.py`：Supervisor LLM advisor，把 LLM 输出转换为 `SupervisorDecision`。
- `llm_decision.py`：LLM decision 统一模型和 validator。限制 forbidden keys、目标引用、score delta、policy/runtime 禁用等。
- `llm_safety.py`：LLM prompt/response 安全工具，负责敏感字段脱敏、响应长度限制和 JSON 深度检查。
- `graph_context.py`：构造发送给 Graph LLM 的紧凑图上下文，避免携带原始工具输出或过长历史；同时暴露 `network_zones`、`reachable_paths`、`pivot_routes`，让 Graph LLM 能区分直连、session、pivot 和内网可达路径。

## `src/core/planner/` 确定性规划内核

该目录是确定性内核，不能 import `src.core.agents.*` 或 LLM client。它只接收模型输入并返回规划/批评结果。

- `planner.py`：AttackGraphPlanner，基于 AG 搜索候选路径并导出 task candidates。
- `scorer.py`：HeuristicScorer 和 scoring context，给 action path/candidate 打分。
- `critic.py`：TaskGraphCritic，确定性分析 TG 的重复任务、低价值任务和 replan frontier。
- `README.md`：规划内核边界规则。

## `src/core/graph/` 图构建与持久化

负责 KG/AG/TG 的存储、投影、初始化、合并和快照。

- `kg_store.py`：内存 KnowledgeGraph 实现，管理 KG 节点/边和查询。
- `ag_projector.py`：KG -> AG 投影，把知识图中的 host/service/goal 等转换为攻击图节点、状态和动作。
- `tg_builder.py`：AG -> TG 构建契约，定义 `TaskCandidate`、`TaskGenerationRequest` 等并生成任务图；新增 worker 相关 action 会映射为端口扫描、Web discovery、凭据验证/复用、横向可达性和内网 fingerprint 任务，并按 host/service/credential/session/route/subnet 生成资源锁键。
- `tg_merge.py`：合并新 TG frontier 到已有 TG，保留生命周期状态。
- `graph_initializer.py`：新 operation 的初始 KG/AG/TG 构建逻辑。
- `topology.py`：KG 网络拓扑查询服务。读取 `CAN_REACH`、`PIVOTS_TO`、`HOSTS`、`BELONGS_TO_ZONE` 边，生成 `ReachabilityPath`、`RouteCandidate`，支持按 source/destination/service 查询可达路径、pivot-only 路径、目标路由候选和 host 所属 zone。
- `graph_memory_store.py`：文件型 Graph Memory Store，保存/加载每个 operation 的 KG、AG、TG、Runtime 快照。

## `src/core/models/` 领域模型

该目录是系统核心数据结构定义，主要用 Pydantic 模型表达 KG/AG/TG/Runtime/Scope/Finding 等。

- `kg.py`：Knowledge Graph 的节点/边模型，如 Host、Service、Goal、DataAsset、关系边等；`CanReachEdge` 记录 source/target host、service、protocol、port、via、route/session，`PivotsToEdge` 记录 source/via/destination、route/session、zone/cidr。
- `kg_enums.py`：KG 使用的枚举，如实体状态。
- `kg_exceptions.py`：KG 相关异常。
- `kg_query.py`：KG 查询模型。
- `kg_types.py`：KG 常用类型别名。
- `ag.py`：Attack Graph 节点、边、图容器和序列化逻辑。
- `tg.py`：Task Graph 节点、边、状态、任务类型、依赖、生命周期模型；任务类型覆盖端口扫描、服务验证、内网服务 fingerprint、Web 枚举/发现、漏洞验证、凭据验证/复用、横向可达性、权限和目标验证。
- `runtime.py`：RuntimeState、OperationRuntime、TaskRuntime、Session、Credential、Budget、ReplanRequest 等运行时模型；`PivotRouteRuntime` 记录 destination host/zone/cidr、allowed ports、protocols、hop count、confidence 和 session 绑定。
- `events.py`：Worker 与运行系统交换的结构化事件、结果、fact write、projection request、runtime request。
- `finding.py`：Finding、Evidence、RiskScore、severity 等发现与报告模型。
- `fingerprint.py`：Recon/fingerprint 输出的规范化模型。
- `scope.py`：目标资产、授权范围、denylist/allowlist 和 engagement scope 模型。
- `vulnerability_candidate.py`：从 fingerprint 推导出的漏洞候选模型。

## `src/core/runtime/` 运行时控制面

Runtime 层管理 operation 的瞬态执行状态、调度、策略、锁、租约、审计、报告和恢复。它不是 KG/AG/TG 的事实源。

- `store.py`：RuntimeState 存储抽象，提供内存和文件存储。
- `scheduler.py`：运行时调度器，结合 TG 与 RuntimeState 做策略受限的任务选择；当任务声明 `require_reachable_route`、`requires_pivot` 或 reachability via 为 `pivot/session` 时，会按目标 host/zone/cidr、端口、协议选择 active route，并把 `route_id`、`selected_route` 写回 task input/runtime metadata；凭据复用和横向可达性任务默认按敏感任务处理。
- `policy.py`：Runtime policy schema、默认策略和加载逻辑；包含 `adapter_policy`、`tool_bindings`、MCP server/tool allowlist 与 denylist 字段。
- `policy_engine.py`：范围、安全和工具策略评估。
- `policy_gate.py`：调度前轻量 policy gate，阻止不允许的 TG 任务进入 worker assignment；凭据复用、横向可达性等敏感 worker 任务默认需要 approval。
- `budgets.py`：成本、风险、噪声等预算控制。
- `locks.py`：资源锁管理。
- `lease_manager.py`：任务与 session 的 lease 管理。
- `session_manager.py`：runtime session 生命周期管理。
- `credential_manager.py`：运行时 credential 状态管理。
- `pivot_route_manager.py`：pivot route 候选、活动路径、服务端口/协议匹配和 session 可用性过滤；支持 `register_candidate`、`refresh_from_reachability`、`select_best_route`、session 失效联动关闭/失败 route。
- `reachability.py`：`ReachabilityPropagator`。把 worker output 中的 `reachability`、`selected_route` 规范化为 Runtime pivot route 更新，并保留 source task、result status、route 详情等元数据。
- `events.py`：runtime event 模型。
- `reducer.py`：把 runtime event 应用到 RuntimeState 的 reducer。
- `result_applier.py`：PhaseTwoResultApplier，消费 worker 结果并通过正确边界写 Runtime/KG/AG/TG/audit；同时处理 `REGISTER_PIVOT_ROUTE`、`VERIFY_PIVOT_ROUTE`、tunnel 和 network namespace runtime request，并委托 `ReachabilityPropagator` 同步 reachability/pivot route 视图。
- `observability.py`：operation log、checkpoint、敏感字段脱敏、clean/unclean shutdown 标记。
- `checkpoint_store.py`：运行时 checkpoint 管理。
- `approvals.py`：人工审批请求模型和状态。
- `runtime_queries.py`：只读 runtime 查询服务。
- `repetition_detector.py`：确定性重复任务检测，避免重复执行已成功/失败/阻塞的任务。
- `risk_scoring.py`：Finding 风险评分。
- `report_generator.py`：Finding/evidence 报告生成。
- `audit_report.py`：operation audit report 汇总，包含 LLM history、control cycle、审计事件等。
- `llm_history.py`：LLM decision history 的存储模型和 append/query helper。

## `src/core/execution/` 工具执行层

Execution 层把 adapter-neutral `ToolPlan` 交给 resolver 和 adapter 执行，并返回 adapter-neutral `ToolExecutionResult`。adapter 不写图、不写 runtime audit。默认选择路径是显式 adapter、policy 强制规则、tool binding、MCP-first、允许的 fallback、fail closed。

- `adapter_resolver.py`：`ToolAdapterResolver`、`AdapterPolicyConfig`、`ToolBinding`。resolver 根据 task type、tool binding、MCP allow/deny、fallback 策略和 adapter 可用性选择最终 adapter；敏感任务可强制 MCP 并禁止本地 fallback。
- `mcp_client.py`：MCP client 协议、`MCPToolCallResult` 和 `UnavailableMCPClient`。当前只定义 tool 调用接口，不接 MCP prompts/resources。
- `pivot_context.py`：把 ToolPlan 中的 route/session hints 与 RuntimeState 中的 `PivotRouteRuntime`、`SessionRuntime` 合并为 adapter-facing execution context，包括 proxy、tunnel endpoint、network namespace、agent ref 和环境变量。
- `tool_plan.py`：从 TG task 构建可执行 ToolPlan。`adapter` 允许为空，新 worker 可以只声明 `tool`、`target`、`args` 和 `metadata.task_type`，由 resolver 选择 MCP 或 fallback。
- `tool_policy.py`：ToolPlan 执行前策略检查，继续复用 `PolicyGate`，并读取 runtime policy 中的 MCP server/tool allowlist 与 denylist。
- `tool_result.py`：执行 adapter 返回的标准结果模型。
- `executor.py`：ExecutionExecutor，通过 `ToolAdapterResolver` 解析最终 adapter，再调用匹配的 `ExecutionAdapter`。显式 legacy adapter 保持兼容，未匹配时 fail closed。
- `__init__.py`：导出 execution API。

### `src/core/execution/adapters/`

- `base.py`：ExecutionAdapter 协议。
- `mcp_adapter.py`：MCPExecutionAdapter。解析 `mcp_server_id`/`mcp_tool_name` binding，调用 MCP client 的 `call_tool`，把 MCP 返回内容转换为 `ToolExecutionResult`，不直接写 KG/AG/TG/Runtime。
- `local_shell_adapter.py`：本地 shell/subprocess 执行 adapter，保留为兼容、调试和低风险 fallback。
- `incalmo_c2_adapter.py`：Incalmo C2 外部执行 adapter。
- `proxy_shell_adapter.py`：SOCKS/HTTP proxy-aware shell adapter，通过 `ALL_PROXY/HTTP_PROXY/HTTPS_PROXY` 等环境变量执行 allowlisted 命令。
- `tunnel_adapter.py`：使用 runtime 中已经打开的 TCP tunnel endpoint，把 endpoint 作为 adapter-neutral 结果返回给后续工具链。
- `netns_shell_adapter.py`：Linux network namespace shell adapter；不支持的平台返回明确 `unsupported_platform`，不隐式降级。
- `__init__.py`：导出 adapter。

## `src/core/perception/` 解析层

Perception parser 把 raw execution/worker result 转成 observation/evidence/fact write request，不负责写 Runtime/KG/AG/TG。

- `parser_protocol.py`：Parser 协议和解析结果模型。
- `parser_registry.py`：按顺序选择 parser，并提供 fallback。
- `generic_parser.py`：通用 fallback parser。
- `tool_execution_parser.py`：解析 adapter-neutral `ToolExecutionResult`。
- `c2_parser.py`：Incalmo C2 parser 的兼容导入 shim，实际逻辑在 integration 目录。
- `__init__.py`：导出 perception API。

## `src/core/feedback/` 反馈辅助

用于从受控工具输出中提取 evidence，并做初步结果校验。

- `evidence_extractor.py`：从 nmap/http 等输出中提取结构化证据。
- `result_verifier.py`：对 controlled tool result 做第一层验证。
- `__init__.py`：包标识。

## `src/core/tools/` 受控工具 recipe

把 LLM/Planner 面向的安全 task hint 转成受控命令或函数调用。自由 shell 命令不应来自模型输出。

- `recipe.py`：ToolTask、ToolRecipe、adapter 和安全转换逻辑。
- `registry.py`：受控 tool recipe 注册表。
- `runner.py`：执行受控 recipe 并记录耗时/结果。
- `__init__.py`：包标识。

## `src/core/vuln_candidates/` 漏洞候选匹配

从 fingerprint 中推导出可能需要验证的漏洞候选。

- `rules.py`：内置候选规则，如 Struts、Redis、Tomcat、Spring Boot 等特征匹配。
- `matcher.py`：CandidateMatcher，根据 service fingerprint 应用规则并输出 `VulnerabilityCandidate`。
- `__init__.py`：导出 matcher/rules。

## `src/core/workers/` Worker 层

Worker 负责执行具体任务并输出结构化结果。Primary worker 实现 `BaseWorkerAgent`，legacy worker 只保留兼容旧协议。默认 worker registry 已覆盖 recon、fingerprint、端口扫描、Web 枚举/发现、access/goal/privilege validation、漏洞验证、凭据验证/复用、横向可达性和内网服务 fingerprint。

- `base.py`：Worker 抽象、能力描述、任务规格、结果封装、异常；`WorkerCapability` 包含 `PORT_SCAN`、`WEB_ENUMERATION`、`WEB_DISCOVERY`、`CREDENTIAL_VALIDATION`、`CREDENTIAL_REUSE_VALIDATION`、`LATERAL_REACHABILITY_VALIDATION`、`INTERNAL_SERVICE_FINGERPRINT` 等受控能力。
- `registry.py`：Primary worker 注册表，默认注册新增 worker、recon/fingerprint/vulnerability worker 和 access/pivot/goal/privilege validation worker；legacy `BaseWorker` 不进入 primary selection。
- `access_validation_worker.py`：访问路径、session、access context 验证 worker。
- `pivot_validation_worker.py`：pivot route probe/health worker，通过已有 session/route 执行探测并输出 reachability、selected_route、tool_execution 和 runtime request。
- `goal_validation_worker.py`：目标状态确认 worker。
- `privilege_validation_worker.py`：权限状态确认 worker。
- `port_scan_worker.py`：受控端口扫描 worker，复用 `ReconWorker`、nmap/masscan/custom adapter 输出 Host/Service、`HOSTS` 关系和后续 fingerprint 候选。
- `web_enum_worker.py`：低风险 Web 枚举 worker，复用 httpx/whatweb/custom adapter 采集 HTTP 状态、标题、header、技术栈等信号。
- `web_discovery_worker.py`：目录/API 发现 worker，只允许同源 GET/HEAD、受 `max_paths`/timeout 约束，输出 `WebEndpoint` 证据。
- `credential_validation_worker.py`：凭据验证 worker，复用 `AccessValidationService` 的 credential validation 边界，输出 credential status 和证据，不直接创建 session。
- `credential_reuse_worker.py`：凭据复用检测 worker，面向受控目标集合输出 reusable 状态，并在成功时给出 lateral reachability 后续候选。
- `lateral_reachability_worker.py`：横向可达性验证 worker，复用 `PivotValidationService` 生成 reachability、selected route 和 `VERIFY_PIVOT_ROUTE` runtime request。
- `internal_service_fingerprint_worker.py`：内网服务 fingerprint worker，要求 route/session 上下文，复用 `FingerprintWorker` 归一化服务 fingerprint 和漏洞候选。
- `fingerprint_worker.py`：服务 fingerprint 归一化，并生成漏洞候选。
- `recon_worker.py`：主机、服务、身份/context 发现 worker，同时兼容大写 TG `TaskType` 的 recon 类任务。
- `vulnerability_validation_worker.py`：受控漏洞验证 worker，调用 registry 中的 validator。
- `probe_adapters.py`：Recon 工具 adapter，如 nmap/httpx/whatweb/custom 的命令构造与输出解析。
- `tool_runner.py`：worker adapter 使用的 subprocess 执行辅助。
- `access_validators.py`：访问验证 adapter。
- `goal_validator.py`：Goal validator 抽象和 metadata fallback validator。
- `goal_command_validator.py`：命令型 goal validator。
- `access_worker.py`：legacy access worker 兼容包装。
- `goal_worker.py`：legacy goal worker 兼容包装。
- `README.md`：worker 边界说明。

### `src/core/workers/services/`

Worker 业务逻辑的单一来源，primary 和 legacy worker 都应复用 service。

- `access_validation_service.py`：access validation 的核心业务逻辑。
- `pivot_validation_service.py`：pivot route validation 的核心业务逻辑，负责 route 选择、可选 probe 执行和 `VERIFY_PIVOT_ROUTE` runtime request 生成。
- `goal_validation_service.py`：goal validation 的核心业务逻辑。
- `privilege_validation_service.py`：privilege validation 的核心业务逻辑。
- `result_builders.py`：worker service 共享结果模型和构造器。
- `__init__.py`：包标识。

### `src/core/workers/vulnerability_validators/`

可插拔漏洞验证器。验证器必须受控、可审计、遵守 scope/policy。

- `base.py`：ValidationTarget、ValidationResult、ValidationStatus、VulnerabilityValidator 抽象。
- `registry.py`：漏洞验证器注册和查找。
- `http_fingerprint.py`：低风险 HTTP fingerprint validator。
- `default_credentials_idor.py`：受控默认凭据/IDOR 证据验证器。
- `struts2_s2045.py`：Struts2 S2-045 validator 插件。
- `__init__.py`：导出 validator 接口和默认实现。

## `src/integrations/` 外部集成

集成外部系统时应保持在 `integrations` 边界内，不让核心 runtime/worker service 直接依赖外部 client。

- `__init__.py`：外部集成包标识。

### `src/integrations/incalmo/`

Incalmo C2 协议集成。

- `models.py`：Incalmo Agent、Command、CommandStatus、CommandResult、LLM action payload 等协议模型。
- `client.py`：Incalmo-compatible C2 HTTP client。
- `mapper.py`：Incalmo 协议对象与 Aegra runtime/result 模型之间的转换。
- `perception.py`：Incalmo C2 command output 的 perception parser 插件。
- `__init__.py`：导出 Incalmo integration API。

## `tests/` 测试目录

测试文件基本按被测模块命名，覆盖面较完整。主要分组如下：

- API 与应用层：`test_api.py`、`test_api_operation_cycle.py`、`test_app_orchestrator.py`、`test_product_api_ui.py`、`test_approval_api.py`、`test_findings_api.py`
- Agent/Pipeline/LLM：`test_agents.py`、`test_planner.py`、`test_critic.py`、`test_supervisor_agent.py`、`test_pipeline_builders.py`、`test_packy_llm.py`、`test_packy_planner_advisor.py`、`test_packy_critic_advisor.py`、`test_graph_llm_planner.py`、`test_llm_decision_models.py`、`test_llm_decision_history.py`
- 图模型与投影：`test_kg` 相关逻辑分散在 `test_graph_initializer.py`、`test_graph_memory_store.py`、`test_ag_models.py`、`test_ag_projector.py`、`test_tg_models.py`、`test_tg_builder.py`、`test_tg_merge.py`、`test_tg_lifecycle_progression.py`
- Runtime/拓扑：`test_runtime_state.py`、`test_runtime_store.py`、`test_runtime_scheduler.py`、`test_runtime_budgets.py`、`test_runtime_locks.py`、`test_runtime_leases.py`、`test_runtime_sessions.py`、`test_runtime_credentials.py`、`test_runtime_events.py`、`test_runtime_checkpoint_store.py`、`test_runtime_pivot_routes.py`、`test_network_topology.py`
- Worker/Execution/Perception：`test_agent_workers.py`、`test_phase_two_workers.py`、`test_phase_two_result_applier.py`、`test_access_service_execution.py`、`test_extended_workers.py`、`test_worker_registry.py`、`test_tool_execution.py`、`test_pivot_execution_context.py`、`test_tool_policy.py`、`test_tool_execution_parser.py`、`test_tool_execution_runtime_audit.py`。其中 `test_tool_execution.py` 覆盖 MCP adapter 结果转换、MCP-first resolver、低风险 fallback、敏感任务 fail closed 和 deny 优先级。
- Fingerprint/漏洞验证：`test_fingerprint_worker.py`、`test_fingerprint_adapters.py`、`test_vulnerability_candidate_matcher.py`、`test_vulnerability_validation_worker.py`、`test_vulnerability_validator_models.py`、`test_vulnerability_validator_registry.py`
- 策略、报告、审计：`test_scope_policy.py`、`test_policy_gate.py`、`test_policy_scheduler_gate.py`、`test_worker_expansion_tg_policy.py`、`test_report_generator.py`、`test_audit_report.py`、`test_risk_scoring.py`
- 架构边界：`test_architecture_boundaries.py`、`test_perception_boundaries.py`
- 外部/烟测：`tests/integration/test_incalmo_c2_live.py`、`test_vulhub_kg_smoke.py`、`test_vulhub_orchestrator_smoke.py`、`test_graph_iteration_vuln_env_smoke.py`

## 关键边界规则

- `src/core/planner` 是确定性内核，不应 import `src.core.agents` 或 LLM client。
- Worker service 不应依赖 Incalmo client。
- Execution adapter 不应 import graph store、ResultApplier 或 KG/AG/TG mutation owner。
- 新增受控 worker 仍必须只输出 `AgentOutput` / `AgentTaskResult` / evidence / outcome / fact-write request / runtime request；不得直接修改 KG、AG、TG 或 RuntimeState。
- 多主机 worker 任务必须显式携带 host/service/credential/session/route/subnet 资源键，凭据复用和横向可达性默认视为敏感动作并进入 approval/policy gate。
- 网络拓扑感知分层：KG 用 `CAN_REACH`、`PIVOTS_TO`、`HOSTS`、`BELONGS_TO_ZONE` 表达事实；`NetworkTopology` 只读查询事实；Runtime 用 `PivotRouteRuntime` 表达当前 operation 的可用路径；Scheduler 只在任务显式需要 route 时选择 active route 并绑定到 task/runtime；Execution resolver 只解析 route/session 为 adapter context；adapter 只执行或暴露 proxy/tunnel/netns 能力；ResultApplier/ReachabilityPropagator 负责把 worker reachability 和 runtime request 落回 RuntimeState。
- Parser 只解析，不写 Runtime/KG/AG/TG。
- ResultApplier 是 worker 结果产生 runtime side effect、KG delta、AG projection、TG lifecycle update 和 audit 的边界。
- LLM 只做受限建议，必须经过 validator，不得直接生成命令、payload、patch 或图写入。
