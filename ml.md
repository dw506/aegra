# Aegra 代码目录与模块职责说明

本文用于快速理解当前代码结构、模块边界和主要文件职责。整体系统围绕 `KG -> AG -> TG -> Scheduler -> Worker -> Execution/MCP -> Perception -> ResultApplier -> Runtime audit` 的闭环组织；当前实验执行路径默认使用单一 `LLMWorkerAgent` 通过 MCP 直接调用工具。

## 顶层目录

```text
D:\Aegra
├── configs/        # MCP 和运行配置示例
│   ├── mcp.lab.json                    # 实验 MCP lab server 配置示例
│   └── mcp.lab.docker.json             # Docker 内运行实验 MCP lab server 的配置示例
├── docs/           # 架构和运行流程文档
│   ├── aegra_runtime_flow.md           # Runtime 主链路、边界和 guardrail 说明
│   └── docker_multihost_lab.md         # 顶层 docker-compose 多目标 smoke 流程说明
├── lab/            # 更完整的 Docker 多网段靶场，含 DMZ/internal/pivot 场景
│   ├── README.md                       # lab 拓扑、启动和 pivot 验证说明
│   ├── docker-compose.yml              # lab 专用 compose，固定 DMZ/internal 地址段
│   ├── Dockerfile.aegra                # lab 中 aegra_agent 镜像
│   ├── scope/                          # lab 授权范围和 policy 示例
│   └── pivot/                          # pivot-ssh 跳板容器构建文件
├── scripts/
│   └── docker_lab_smoke.ps1            # 顶层 Docker lab smoke 自动化脚本
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
│   ├── integrations/                   # 实验 MCP server 集成
│   │   └── mcp_lab/                    # 隔离实验环境 MCP server 和 v1 lab tools
│   ├── core/                           # 核心领域逻辑
│   │   ├── actions/                    # AG action 到 TG task 的确定性模板
│   │   │   ├── __init__.py             # action template API 导出
│   │   │   └── schemas.py              # ActionTemplate/Input/Output 和任务构造逻辑
│   │   ├── agents/                     # Agent 协议、pipeline、planner、critic、LLM advisor
│   │   │   ├── agent_protocol.py       # AgentInput/Output、权限、上下文和 BaseAgent
│   │   │   ├── agent_models.py         # Agent 层共享 record 模型
│   │   │   ├── registry.py             # Agent 注册、查找和 dispatch
│   │   │   ├── agent_pipeline.py       # Planner/Builder/Scheduler/Worker/Feedback/Supervisor 串联
│   │   │   ├── pipeline_builders.py    # 可选 pipeline 装配、LLM advisor 和默认 LLM worker 注入
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
│   │   ├── execution/                  # ToolPlan 执行、MCP-first resolver、adapter dispatch 和实验 MCP client
│   │   │   ├── __init__.py             # execution API 导出
│   │   │   ├── adapter_resolver.py     # ToolAdapterResolver，按 policy/binding/MCP-first/fallback 选择 adapter
│   │   │   ├── mcp_client.py           # MCP client 协议、tool catalog、标准 MCP tool call result 和 unavailable 默认实现
│   │   │   ├── configured_mcp_client.py # 从 MCP 配置装配 stdio/http client，支持持久 stdio session、tools/list 和 tools/call
│   │   │   ├── pivot_context.py        # pivot route/session -> adapter execution context
│   │   │   ├── tool_plan.py            # TG task -> adapter-neutral ToolPlan，adapter 可为空
│   │   │   ├── tool_policy.py          # ToolPlan 执行前策略检查，含 MCP allow/deny 检查
│   │   │   ├── tool_result.py          # adapter-neutral 执行结果模型
│   │   │   ├── executor.py             # ExecutionExecutor，通过 resolver 选择 adapter 并执行
│   │   │   └── adapters/               # 执行 adapter
│   │   │       ├── __init__.py         # adapter API 导出
│   │   │       ├── base.py             # ExecutionAdapter 协议
│   │   │       ├── mcp_adapter.py      # MCPExecutionAdapter，调用 MCP tool 并转换为 ToolExecutionResult
│   │   │       ├── local_shell_adapter.py  # 本地 shell/subprocess adapter，支持 argv/env/timeout/截断/exit code 归一化
│   │   │       ├── http_request_adapter.py # 受控 HTTP GET/HEAD execution adapter
│   │   │       ├── proxy_shell_adapter.py  # SOCKS/HTTP proxy 环境注入 adapter
│   │   │       ├── tunnel_adapter.py       # 预开 TCP tunnel endpoint adapter
│   │   │       └── netns_shell_adapter.py  # Linux network namespace shell adapter
│   │   ├── perception/                 # raw result -> observation/evidence parser
│   │   │   ├── __init__.py             # perception API 导出
│   │   │   ├── parser_protocol.py      # parser 协议和结果模型
│   │   │   ├── parser_registry.py      # parser registry 和 fallback 选择
│   │   │   ├── generic_parser.py       # 通用 fallback parser
│   │   │   └── tool_execution_parser.py # ToolExecutionResult parser
│   │   ├── feedback/                   # evidence 提取和结果校验
│   │   │   ├── __init__.py             # 包标识
│   │   │   ├── evidence_extractor.py   # 从受控工具输出提取证据
│   │   │   └── result_verifier.py      # controlled result 初步验证
│   │   ├── tools/                      # 受控 tool recipe
│   │   │   ├── __init__.py             # 包标识
│   │   │   ├── recipe.py               # ToolTask/ToolRecipe 和安全转换
│   │   │   ├── registry.py             # recipe registry
│   │   │   └── runner.py               # recipe 执行器
│   │   ├── visualization/              # KG/AG/TG/Runtime 可视化读模型和事件发布
│   │   │   ├── __init__.py             # visualization API 导出
│   │   │   ├── graph_event.py          # VisualGraphSnapshot/Delta/Node/Edge 模型
│   │   │   ├── graph_serializer.py     # 图和 RuntimeState -> dashboard read model
│   │   │   └── graph_publisher.py      # operation 级别 WebSocket delta publisher
│   │   ├── vuln_candidates/            # fingerprint -> 漏洞候选
│   │   │   ├── __init__.py             # matcher/rules 导出
│   │   │   ├── rules.py                # 内置候选规则
│   │   │   └── matcher.py              # CandidateMatcher
│   │   └── workers/                    # 默认 LLM worker、legacy/受控 worker、domain service
│   │       ├── README.md               # worker 边界说明
│   │       ├── base.py                 # BaseWorkerAgent、任务规格、结果封装
│   │       ├── registry.py             # primary worker registry，默认只注册 LLMWorkerAgent
│   │       ├── llm_worker.py           # 实验 LLM worker，接收 scheduler task 并直接调用 MCP tool
│   │       ├── llm_worker_advisor.py   # LLM worker advisor，生成结构化 MCP 调用 JSON
│   │       ├── llm_worker_models.py    # LLMWorkerDecision 输出模型
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
├── tests/          # 单元、集成、烟测测试
│   ├── integration/                     # 需要外部服务/显式 env 开关的集成测试
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
│   ├── test_llm_worker.py               # LLM worker/advisor/MCP config/pipeline 分发测试
│   ├── test_mcp_lab.py                  # 实验 MCP lab server、v1 tool catalog 和 stdio client 测试
│   ├── test_extended_workers.py         # 新增受控 worker 行为测试
│   ├── test_worker_expansion_tg_policy.py # 新 worker 的 TG 映射/资源锁/policy 测试
│   ├── test_tool_execution.py           # ToolPlan/executor/MCP/local shell/http request adapter/resolver 测试
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
│   ├── test_architecture_boundaries.py  # 架构依赖边界测试
│   ├── test_perception_boundaries.py    # perception 边界测试
│   ├── test_nuclei_safe_adapter.py      # nuclei safe adapter 测试
│   ├── test_scorer.py                   # planner scorer 测试
│   ├── test_findings.py                 # Finding 模型/逻辑测试
│   ├── test_vulhub_kg_smoke.py          # Vulhub KG smoke 测试
│   ├── test_vulhub_orchestrator_smoke.py # Vulhub orchestrator smoke 测试
│   └── test_graph_iteration_vuln_env_smoke.py # 图迭代漏洞环境 smoke 测试
├── var/            # 本地运行态、审计、图快照等生成数据
├── web/
│   └── dashboard/                      # Vite/React 图看板，消费 visual-graphs API/WS
│       ├── src/App.tsx                 # operation 选择、KG/AG/TG/Runtime tabs、过滤和导出
│       ├── src/api.ts                  # dashboard API/WS URL 封装
│       ├── src/graphState.ts           # snapshot/delta -> 前端图状态
│       └── src/components/             # Cytoscape 图、TG 图和节点详情面板
├── Dockerfile                          # 顶层 Aegra API 镜像
├── docker-compose.yml                  # API、dashboard、三台 target 和 internal-service smoke 拓扑
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
  -> LLMWorkerAgent
  -> LLMWorkerAdvisor
  -> MCPClient.call_tool
  -> MCPToolCallResult / tool_execution payload
  -> Perception / ParserRegistry
  -> PhaseTwoResultApplier / ReachabilityPropagator
  -> RuntimeState + KG/AG/TG 更新 + 审计 + VisualGraphDelta
```

LLM 通过 advisor 层进入系统：

- Planner rank advisor：对已有候选排序和解释增强。
- Graph LLM planner advisor：基于图上下文提出声明式 task proposal；图上下文包含 KG reachability path、network zone 和 Runtime pivot route 摘要。
- Critic advisor：总结失败原因和 replan hint。
- Supervisor advisor：在有限策略集合中给高层控制建议。
- 实验 LLM worker advisor：读取 scheduler 任务输入和 MCP tool catalog，返回 `call_mcp_tool/defer/failed` 的结构化 JSON；当前实验模式不做 ToolIntentValidator/PolicyGate，只做 JSON schema 解析。

默认 worker 执行路径允许 LLM 在隔离实验环境中选择 MCP server/tool/arguments，由 MCP server 负责具体命令执行。除该实验路径外，Planner/Critic/Supervisor 仍不得直接执行命令、生成 shell payload、写 KG/AG/TG/Runtime。

## `configs/`

- `mcp.lab.json`：本机 MCP 配置。包含内置 `pentest-tools` stdio server，以及可选的 Docker `pentest-mcp` stdio server。
- `mcp.lab.docker.json`：容器内 MCP 配置。包含 `/app` 下运行的内置 `pentest-tools` stdio server，以及指向 `http://host.docker.internal:8765/mcp` 的 `external-tools` HTTP server。

## `docs/`

- `aegra_runtime_flow.md`：当前运行闭环、所有权边界和 guardrail 说明。
- `docker_multihost_lab.md`：顶层 Docker Compose smoke 拓扑和多目标运行命令说明。

## `lab/` Docker 多网段实验环境

`lab/` 是独立于顶层 compose smoke 的更完整靶场。它用固定网段表达 DMZ、internal-only 网络和 SSH pivot 场景，适合验证 route/session/pivot 相关能力。

- `README.md`：记录 `aegra_agent`、DMZ vulnerable targets、`pivot-ssh`、internal web/db 的地址和验证命令。
- `docker-compose.yml`：lab 专用编排文件，创建 DMZ/internal 网络、Aegra agent、DVWA、Juice Shop、Vulhub S2-045、pivot 和内部服务。
- `Dockerfile.aegra`：lab 中 Aegra agent 镜像。
- `scope/docker_lab.yaml`：lab 授权范围示例。
- `scope/docker_lab.policy.json`：lab runtime policy 示例。
- `pivot/Dockerfile`：SSH pivot 容器构建文件。
- `outputs/`：lab 运行态和审计输出目录，当前只保留 `.gitignore`。

## `web/dashboard/` React 图看板

`web/dashboard/` 是 Vite/React 前端，用于查看 operation 的 KG、AG、TG 和 Runtime 可视化读模型。它通过 REST 拉取初始 snapshot，通过 WebSocket 接收增量 delta。

- `src/App.tsx`：主界面。列出 operations，按 KG/AG/TG/Runtime 分 tab 展示图，支持 type/status 过滤、刷新、连接状态和 JSON 导出。
- `src/api.ts`：封装 `/operations`、`/operations/{id}/visual-graphs/snapshot` 和 `/operations/{id}/visual-graphs/ws`，支持 `VITE_GRAPH_API_BASE`、`VITE_GRAPH_WS_BASE`。
- `src/graphState.ts`：把后端 `VisualGraphSnapshot`/`VisualGraphDelta` 转成前端节点/边 map，并处理 upsert/delete/update_status。
- `src/components/CytoscapeGraph.tsx`：KG/AG/Runtime 的 Cytoscape 图渲染。
- `src/components/TaskGraph.tsx`：TG 专用图渲染。
- `src/components/NodeDetailPanel.tsx`：选中节点详情面板。
- `src/types.ts`：前端消费的 operation、graph snapshot、delta、node/edge 类型。
- `src/graphState.test.ts`：前端图状态转换测试。
- `Dockerfile`、`package.json`、`vite.config.ts`、`tsconfig.json`：dashboard 构建和开发环境配置。

## 顶层 Docker 与脚本

- `Dockerfile`：构建 Aegra FastAPI 控制面镜像，设置 `/app` 工作目录并安装 Python 依赖。
- `docker-compose.yml`：顶层 smoke 拓扑。包含 `aegra` API、`dashboard` 前端、三台 Web target 和一个只在 internal network 暴露的 `internal-service`。
- `.dockerignore`：Docker build context 排除规则。
- `scripts/docker_lab_smoke.ps1`：自动执行顶层 Docker lab smoke；默认结束后清理容器，`-KeepRunning` 可保留环境。

## `src/app/` 应用层

应用层负责把核心能力装配成可运行的本地控制面：配置、API、operation 生命周期、运行主循环、审计查询和前端静态资源。

- `settings.py`：环境变量和运行配置入口。包含 runtime store、审计、策略、LLM、MCP、工具路径等配置。MCP 相关字段包括 `mcp_enabled`、`mcp_first`、`mcp_config_path`、`mcp_config_json`、`mcp_default_timeout_seconds` 和 `allow_local_fallback`，并支持 `AEGRA_MCP_ENABLED`、`AEGRA_MCP_FIRST`、`AEGRA_MCP_CONFIG_PATH`、`AEGRA_MCP_CONFIG_JSON`、`AEGRA_MCP_DEFAULT_TIMEOUT_SECONDS`、`AEGRA_ALLOW_LOCAL_FALLBACK`；默认 pipeline 会把这些配置用于实验 `LLMWorkerAgent` 的 MCP client 装配。
- `orchestrator.py`：应用级编排门面。负责 operation 创建/启动/停止、目标导入、运行 control cycle、加载/保存图记忆、调用 pipeline、应用 worker 结果、发布 visual graph delta、记录审计和控制摘要。
- `llm_decision_observer.py`：从 pipeline cycle 输出中提取 prompt-free 的 LLM decision history，负责记录 accepted/rejected、decision id、target id、模型和 usage 等审计摘要。

### `src/app/api/`

API 层负责暴露本地控制面 HTTP 接口。

- `__init__.py`：FastAPI 应用主入口和当前主要 API surface，注册 operation、approval、findings、audit、LLM history 等路由。
- `operation_routes.py`：operation lifecycle 路由边界占位，当前大部分仍由 `__init__.py` 兼容注册。
- `execution_routes.py`：execution/approval 路由边界占位。
- `graph_routes.py`：图可视化 read-model 路由。提供 `/operations/{operation_id}/visual-graphs/snapshot` 初始快照和 `/operations/{operation_id}/visual-graphs/ws` WebSocket 增量订阅。

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
- `pipeline_builders.py`：可选 pipeline 装配逻辑，根据配置启用 Packy/Graph LLM advisor、critic advisor、supervisor advisor；默认同时注册单一 `LLMWorkerAgent`。

### 规划与任务生成

- `planner.py`：PlannerAgent。读取 AG/goal/graph_context，生成 planning candidates；可选消费 LLM 排序建议和 Graph LLM task proposal，但不写 TG、不执行工具。
- `task_builder.py`：TaskBuilderAgent。把 planner 选中的 candidate/action chain 转成 TG patch delta。
- `scheduler_agent.py`：SchedulerAgent。读取 TG 和 RuntimeState，找出可调度任务并生成 worker assignment decisions；实际 route-aware admissibility 由 `RuntimeScheduler` 执行。当前实验默认下 assignment 仍保留，但 pipeline 最终只会路由到唯一 `LLMWorkerAgent`。

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
- `workers/llm_worker_advisor.py`：实验 worker 的 LLM advisor，复用 `PackyLLMClient`，让 LLM 基于 scheduler task、graph refs 和 MCP tool catalog 选择 MCP 工具调用。

## `src/integrations/` 外部集成层

Integration 层封装实验 server，不拥有 KG/AG/TG/Runtime 写权限。当前包含实验 MCP lab server。

- `mcp_lab/http_server.py`：实验 MCP lab HTTP JSON-RPC server，监听 `/mcp`，复用 stdio server 的 request handler。
- `mcp_lab/server.py`：隔离实验环境 MCP stdio server，提供 JSON-RPC `initialize`、`tools/list`、`tools/call`。
- `mcp_lab/tools.py`：v1 lab tools：`run_command`、`nmap_scan`、`http_probe`、`web_fingerprint`、`web_discover`、`dns_lookup`、`tls_probe`。所有工具执行前都要求 `AEGRA_LAB_MODE=1`。
- `mcp_lab` 工具统一返回 `stdout/stderr/exit_code/parsed/artifacts`，其中 `parsed` 包含 `entities`、`relations`、`findings`、`writeback_hints` 和 `runtime_hints`，由 LLM worker 透传给 perception/result applier。

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
- `result_applier.py`：PhaseTwoResultApplier，消费 worker 结果并通过正确边界写 Runtime/KG/AG/TG/audit；同时处理 `REGISTER_PIVOT_ROUTE`、`VERIFY_PIVOT_ROUTE`、tunnel 和 network namespace runtime request，委托 `ReachabilityPropagator` 同步 reachability/pivot route 视图，并产出 KG/AG/TG/Runtime 的 `VisualGraphDelta` 供 WebSocket 发布。
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

Execution 层保留两条路径：传统路径把 adapter-neutral `ToolPlan` 交给 resolver 和 adapter 执行，并返回 adapter-neutral `ToolExecutionResult`；实验 LLM worker 路径绕过 `ToolPlan/ExecutionExecutor/MCPExecutionAdapter`，直接使用 `MCPClient.call_tool`，再把结果包装成 `tool_execution` payload。adapter/client 不写图、不写 runtime audit。

- `adapter_resolver.py`：`ToolAdapterResolver`、`AdapterPolicyConfig`、`ToolBinding`。resolver 根据 task type、tool binding、MCP allow/deny、fallback 策略和 adapter 可用性选择最终 adapter；敏感任务可强制 MCP 并禁止本地 fallback。
- `mcp_client.py`：MCP client 协议、`list_tools`、`MCPToolCallResult` 和 `UnavailableMCPClient`。
- `configured_mcp_client.py`：配置驱动 MCP client，从 path/json 装配 stdio/http server，支持 HTTP JSON-RPC、持久 stdio session、`tools/list` 缓存、`tools/call` 归一化、timeout 和 JSON-RPC 错误归一化。
- `pivot_context.py`：把 ToolPlan 中的 route/session hints 与 RuntimeState 中的 `PivotRouteRuntime`、`SessionRuntime` 合并为 adapter-facing execution context，包括 proxy、tunnel endpoint、network namespace、agent ref 和环境变量。
- `tool_plan.py`：从 TG task 构建可执行 ToolPlan。`adapter` 允许为空，传统受控 worker 可以只声明 `tool`、`target`、`args` 和 `metadata.task_type`，由 resolver 选择 MCP 或 fallback；实验 LLM worker 当前不使用该模型。
- `tool_policy.py`：ToolPlan 执行前策略检查，继续复用 `PolicyGate`，并读取 runtime policy 中的 MCP server/tool allowlist 与 denylist。
- `tool_result.py`：执行 adapter 返回的标准结果模型。
- `executor.py`：ExecutionExecutor，通过 `ToolAdapterResolver` 解析最终 adapter，再调用匹配的 `ExecutionAdapter`。显式 legacy adapter 保持兼容，未匹配时 fail closed。
- `__init__.py`：导出 execution API。

### `src/core/execution/adapters/`

- `base.py`：ExecutionAdapter 协议。
- `mcp_adapter.py`：MCPExecutionAdapter。解析 `mcp_server_id`/`mcp_tool_name` binding，调用 MCP client 的 `call_tool`，把 MCP 返回内容转换为 `ToolExecutionResult`，不直接写 KG/AG/TG/Runtime。
- `local_shell_adapter.py`：本地 shell/subprocess 执行 adapter，支持 `args.argv`、cwd/env/env allowlist、command allowlist、acceptable exit codes、timeout、stdout/stderr 截断和 `success/nonzero_exit/timeout/command_not_found/process_error/policy_denied` 分类；保留为兼容、调试和低风险 fallback。
- `http_request_adapter.py`：受控 HTTP request adapter，执行同源约束下的 GET/HEAD 请求，把 status、headers、content type、auth_required、reachable、blocked_reason 等归一化到 `ToolExecutionResult.metadata`。
- `proxy_shell_adapter.py`：SOCKS/HTTP proxy-aware shell adapter，通过 `ALL_PROXY/HTTP_PROXY/HTTPS_PROXY` 等环境变量执行 allowlisted 命令。
- `tunnel_adapter.py`：使用 runtime 中已经打开的 TCP tunnel endpoint，把 endpoint 作为 adapter-neutral 结果返回给后续工具链。
- `netns_shell_adapter.py`：Linux network namespace shell adapter；不支持的平台返回明确 `unsupported_platform`，不隐式降级。
- `__init__.py`：导出 adapter。

### 实验 MCP lab server 启动方式

实验执行路径默认由 `LLMWorkerAgent` 调用 MCP。最小启动配置：

```powershell
$env:AEGRA_MCP_ENABLED = "true"
$env:AEGRA_MCP_CONFIG_PATH = "D:\Aegra\configs\mcp.lab.json"
```

`configs/mcp.lab.json` 使用 stdio transport 启动 `python -m src.integrations.mcp_lab.server`，并给 server 注入 `AEGRA_LAB_MODE=1`。LLM worker 会先通过 `tools/list` 读取工具目录，再按 LLM JSON 决策调用 `tools/call`。MCP tool result 会被归一化到 `AgentOutput.outcomes[].payload.tool_execution` 和 `evidence[].extra.parsed`。

## `src/core/perception/` 解析层

Perception parser 把 raw execution/worker result 转成 observation/evidence/fact write request，不负责写 Runtime/KG/AG/TG。

- `parser_protocol.py`：Parser 协议和解析结果模型。
- `parser_registry.py`：按顺序选择 parser，并提供 fallback。
- `generic_parser.py`：通用 fallback parser。
- `tool_execution_parser.py`：解析 adapter-neutral `ToolExecutionResult` 和实验 LLM worker 的 `mcp_direct` payload；当 MCP 返回 `parsed` 时，会保留 entities、relations、findings、runtime hints 和 writeback hints。
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

## `src/core/visualization/` 图可视化读模型

Visualization 层把 KG/AG/TG 和 RuntimeState 转成只读 dashboard 模型，并发布 operation 级别的增量事件。它是 read-model/export 边界，不应被 parser、worker 或 execution adapter 反向依赖。

- `graph_event.py`：定义 `VisualGraphSnapshot`、`VisualGraphDelta`、`VisualGraphState`、`VisualNode`、`VisualEdge` 和 `GraphOperation`。
- `graph_serializer.py`：把 KG/AG/TG payload 和 RuntimeState 序列化为 dashboard 可消费的节点/边状态，也能把图 payload/runtime 转成 delta。
- `graph_publisher.py`：进程内 `GraphDeltaPublisher`，按 operation id 维护 WebSocket subscriber queue，并在队列满时丢弃最旧 delta。
- `__init__.py`：导出 visualization API。

## `src/core/vuln_candidates/` 漏洞候选匹配

从 fingerprint 中推导出可能需要验证的漏洞候选。

- `rules.py`：内置候选规则，如 Struts、Redis、Tomcat、Spring Boot 等特征匹配。
- `matcher.py`：CandidateMatcher，根据 service fingerprint 应用规则并输出 `VulnerabilityCandidate`。
- `__init__.py`：导出 matcher/rules。

## `src/core/workers/` Worker 层

Worker 负责执行具体任务并输出结构化结果。Primary worker 实现 `BaseWorkerAgent`，legacy worker 只保留兼容旧协议。当前默认 worker registry 只注册 `LLMWorkerAgent`：Scheduler 仍输出 assignment decision，pipeline 将任务转成 worker input，LLM worker 再让 LLM 选择 MCP server/tool/arguments 并直接调用 MCP。原 recon、fingerprint、端口扫描、Web 枚举/发现、access/goal/privilege validation、漏洞验证、凭据验证/复用、横向可达性和内网服务 fingerprint worker 仍保留为可显式注册的受控/兼容实现。

- `base.py`：Worker 抽象、能力描述、任务规格、结果封装、异常；`WorkerCapability` 包含 `PORT_SCAN`、`WEB_ENUMERATION`、`WEB_DISCOVERY`、`CREDENTIAL_VALIDATION`、`CREDENTIAL_REUSE_VALIDATION`、`LATERAL_REACHABILITY_VALIDATION`、`INTERNAL_SERVICE_FINGERPRINT` 等受控能力。
- `registry.py`：Primary worker 注册表，默认只注册 `LLMWorkerAgent`；legacy `BaseWorker` 不进入 primary selection。
- `llm_worker.py`：实验单一 worker。`supports_task()` 接收所有任务，调用 `LLMWorkerAdvisor` 获取 MCP 调用决策，直接执行 `MCPClient.call_tool`，并输出 `tool_execution`、`llm_mcp_decision` 和 writeback hints。
- `llm_worker_advisor.py`：实验 worker advisor。构造包含 task、graph refs、raw payload 和 MCP tool catalog 的 prompt，要求 LLM 只返回 JSON，并解析成 `LLMWorkerDecision`。
- `llm_worker_models.py`：`LLMWorkerDecision`，字段包括 `action`、`server_id`、`tool_name`、`arguments`、`summary`、`expected_evidence`、`risk_assessment`、`writeback_hints`。
- `access_validation_worker.py`：访问路径、session、access context 验证 worker。
- `pivot_validation_worker.py`：pivot route probe/health worker，通过已有 session/route 执行探测并输出 reachability、selected_route、tool_execution 和 runtime request。
- `goal_validation_worker.py`：目标状态确认 worker。
- `privilege_validation_worker.py`：权限状态确认 worker。
- `port_scan_worker.py`：受控端口扫描 worker，复用 `ReconWorker`，由 `ReconWorker` 构造 `ToolPlan` 并通过 `ExecutionExecutor` 调用 nmap/masscan/custom 等 probe 命令，输出 Host/Service、`HOSTS` 关系和后续 fingerprint 候选。
- `web_enum_worker.py`：低风险 Web 枚举 worker，复用 httpx/whatweb/custom adapter 采集 HTTP 状态、标题、header、技术栈等信号。
- `web_discovery_worker.py`：目录/API 发现 worker，只生成同源 GET/HEAD 探测计划，HTTP I/O 通过 `ExecutionExecutor` 和 `HttpRequestExecutionAdapter` 执行；受 `max_paths`/timeout 约束，输出 `WebEndpoint` 证据。
- `credential_validation_worker.py`：凭据验证 worker，复用 `AccessValidationService` 的 credential validation 边界，输出 credential status 和证据，不直接创建 session。
- `credential_reuse_worker.py`：凭据复用检测 worker，面向受控目标集合输出 reusable 状态，并在成功时给出 lateral reachability 后续候选。
- `lateral_reachability_worker.py`：横向可达性验证 worker，复用 `PivotValidationService` 生成 reachability、selected route 和 `VERIFY_PIVOT_ROUTE` runtime request。
- `internal_service_fingerprint_worker.py`：内网服务 fingerprint worker，要求 route/session 上下文，复用 `FingerprintWorker` 归一化服务 fingerprint 和漏洞候选。
- `fingerprint_worker.py`：服务 fingerprint 归一化，并生成漏洞候选。
- `recon_worker.py`：主机、服务、身份/context 发现 worker，同时兼容大写 TG `TaskType` 的 recon 类任务；负责选择 `ProbeAdapter`、构造 adapter-neutral `ToolPlan`、调用 `ExecutionExecutor`，再把执行结果交给 probe parser 生成结构化 evidence。
- `vulnerability_validation_worker.py`：受控漏洞验证 worker，调用 registry 中的 validator。
- `probe_adapters.py`：Recon 工具 adapter，如 nmap/httpx/whatweb/custom 的命令构造与输出解析。
- `tool_runner.py`：legacy subprocess 执行辅助，仍供旧 validator/compat 路径使用；primary recon/web discovery 外部 I/O 不再直接依赖它。
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

## `tests/` 测试目录

测试文件基本按被测模块命名，覆盖面较完整。主要分组如下：

- API 与应用层：`test_api.py`、`test_api_operation_cycle.py`、`test_app_orchestrator.py`、`test_product_api_ui.py`、`test_approval_api.py`、`test_findings_api.py`
- Agent/Pipeline/LLM：`test_agents.py`、`test_planner.py`、`test_critic.py`、`test_supervisor_agent.py`、`test_pipeline_builders.py`、`test_packy_llm.py`、`test_packy_planner_advisor.py`、`test_packy_critic_advisor.py`、`test_graph_llm_planner.py`、`test_llm_decision_models.py`、`test_llm_decision_history.py`
- 图模型与投影：`test_kg` 相关逻辑分散在 `test_graph_initializer.py`、`test_graph_memory_store.py`、`test_ag_models.py`、`test_ag_projector.py`、`test_tg_models.py`、`test_tg_builder.py`、`test_tg_merge.py`、`test_tg_lifecycle_progression.py`
- Runtime/拓扑：`test_runtime_state.py`、`test_runtime_store.py`、`test_runtime_scheduler.py`、`test_runtime_budgets.py`、`test_runtime_locks.py`、`test_runtime_leases.py`、`test_runtime_sessions.py`、`test_runtime_credentials.py`、`test_runtime_events.py`、`test_runtime_checkpoint_store.py`、`test_runtime_pivot_routes.py`、`test_network_topology.py`
- Worker/Execution/Perception：`test_agent_workers.py`、`test_phase_two_workers.py`、`test_phase_two_result_applier.py`、`test_access_service_execution.py`、`test_extended_workers.py`、`test_worker_registry.py`、`test_llm_worker.py`、`test_tool_execution.py`、`test_pivot_execution_context.py`、`test_tool_policy.py`、`test_tool_execution_parser.py`、`test_tool_execution_runtime_audit.py`。其中 `test_llm_worker.py` 覆盖 LLM worker JSON 决策解析、直接 MCP 调用、defer/failed 分支、scheduler 到唯一 worker 的分发和 MCP config 装配；`test_tool_execution.py` 覆盖传统 MCP adapter 结果转换、MCP-first resolver、本地 shell 执行归一化、HTTP request adapter、低风险 fallback、敏感任务 fail closed 和 deny 优先级。
- Visualization/看板：`test_visualization.py` 覆盖 KG/AG/TG/Runtime snapshot 序列化、snapshot API 和 WebSocket delta；`web/dashboard/src/graphState.test.ts` 覆盖前端 snapshot/delta 状态转换。
- Fingerprint/漏洞验证：`test_fingerprint_worker.py`、`test_fingerprint_adapters.py`、`test_vulnerability_candidate_matcher.py`、`test_vulnerability_validation_worker.py`、`test_vulnerability_validator_models.py`、`test_vulnerability_validator_registry.py`
- 策略、报告、审计：`test_scope_policy.py`、`test_policy_gate.py`、`test_policy_scheduler_gate.py`、`test_worker_expansion_tg_policy.py`、`test_report_generator.py`、`test_audit_report.py`、`test_risk_scoring.py`
- 架构边界：`test_architecture_boundaries.py`、`test_perception_boundaries.py`
- 外部/烟测：`test_vulhub_kg_smoke.py`、`test_vulhub_orchestrator_smoke.py`、`test_graph_iteration_vuln_env_smoke.py`

## 关键边界规则

- `src/core/planner` 是确定性内核，不应 import `src.core.agents` 或 LLM client。
- Worker service 不应依赖外部控制通道 client。
- Execution adapter 不应 import graph store、ResultApplier 或 KG/AG/TG mutation owner。
- Visualization 是只读模型和事件发布边界；parser、worker、execution 层不应 import `src.core.visualization`。
- Primary Recon/WebDiscovery worker 不应直接调用 legacy `ToolRunner` 或 `urllib.request.urlopen` 执行外部 I/O，必须构造 `ToolPlan` 并交给 `ExecutionExecutor`；实验 `LLMWorkerAgent` 是例外，它直接调用 `MCPClient.call_tool` 并输出 `tool_execution` payload。
- 新增受控 worker 仍必须只输出 `AgentOutput` / `AgentTaskResult` / evidence / outcome / fact-write request / runtime request；不得直接修改 KG、AG、TG 或 RuntimeState。
- 多主机 worker 任务必须显式携带 host/service/credential/session/route/subnet 资源键，凭据复用和横向可达性默认视为敏感动作并进入 approval/policy gate。
- 网络拓扑感知分层：KG 用 `CAN_REACH`、`PIVOTS_TO`、`HOSTS`、`BELONGS_TO_ZONE` 表达事实；`NetworkTopology` 只读查询事实；Runtime 用 `PivotRouteRuntime` 表达当前 operation 的可用路径；Scheduler 只在任务显式需要 route 时选择 active route 并绑定到 task/runtime；Execution resolver 只解析 route/session 为 adapter context；adapter 只执行或暴露 proxy/tunnel/netns 能力；ResultApplier/ReachabilityPropagator 负责把 worker reachability 和 runtime request 落回 RuntimeState。
- Parser 只解析，不写 Runtime/KG/AG/TG。
- ResultApplier 是 worker 结果产生 runtime side effect、KG delta、AG projection、TG lifecycle update 和 audit 的边界。
- Planner/Critic/Supervisor LLM 只做受限建议，必须经过对应 validator，不得直接生成命令、payload、patch 或图写入。实验 `LLMWorkerAgent` 在隔离环境中允许 LLM 选择 MCP 工具和参数，当前只做 JSON schema 解析，不做工具安全决策。
