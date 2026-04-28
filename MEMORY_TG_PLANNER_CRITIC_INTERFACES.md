# TG / Planner / Critic / Runtime / Agent 记忆文档

日期: 2026-04-14

本文档记录当前仓库中与 KG / AG / Planner / TG / Runtime / Agent 相关的接口边界、实现约定、已落地模块和测试状态。

## 1. 总体架构

当前主线框架可概括为：

- `Knowledge Graph (KG) -> Attack Graph (AG) -> Planner -> Task Graph (TG) -> Runtime State -> Agent / Worker`

当前 agent 化编排主链可概括为：

- `Planner -> TaskBuilder -> Scheduler -> Worker -> Perception -> StateWriter -> GraphProjection -> Critic`

职责分层：

- KG：世界事实与证据层。
- AG：可规划状态、动作空间与约束层。
- Planner：只输出 decision / candidate，不执行工具。
- TG：执行任务结构，只保留最小调度字段与局部恢复结构。
- Runtime State：执行现场、锁、session、预算、checkpoint、事件。
- Agent / Worker：按统一协议消费输入、产出结构化结果，不直接改 KG / AG / TG 主结构。
- AgentPipeline：只负责 cycle 编排、转发和聚合，不替代各 agent 的 owner 职责。

## 2. 关键边界

当前实现遵守以下边界：

- Runtime State 不保存世界事实，不替代 KG。
- Runtime State 不保存理论动作空间，不替代 AG。
- Runtime State 不重建任务结构，不替代 TG。
- Worker 不直接写 KG / AG / TG 主结构。
- Perception Agent 只把 worker 输出转换为 observation / evidence，不直接写 KG / AG / TG。
- State Writer 是唯一正式写入 KG 事实层的 agent。
- Graph Projection 是唯一正式构建或更新 AG 的 agent。
- Task Builder / Scheduler 是 TG 的主要拥有者。
- Planner 只能产出 decision / task candidates，不能直接执行工具。
- Critic 只能产出 cancel / replace / replan 建议，不直接改事实层。
- 运行时事件只存在于 Runtime 层，不写入 KG / AG / TG。

## 3. 现有目录职责

- `src/core/models`
  定义 AG / KG / TG / Runtime 相关核心模型。
- `src/core/graph`
  定义 KG store、AG projector、TG builder。
- `src/core/planner`
  包含 planner、scorer、critic。
- `src/core/runtime`
  包含 events、reducer、store、locks、sessions、budgets、checkpoint、queries、scheduler。
- `src/core/workers`
  包含 worker 基础抽象与具体 worker 实现。
- `src/core/agents`
  新增的 agent 协议层、共享模型层、owner agents、registry 与 pipeline。
- `tests`
  覆盖 Planner / TG / Critic / Runtime / Agent 相关单测。

## 4. TG / Planner / Critic 约定

### 4.1 TG 最小调度字段

TG 保留并面向 scheduler 提供的最小字段包括：

- `resource_keys`
- `parallelizable`
- `status`
- `assigned_agent`
- `attempt_count`
- `max_attempts`
- `deadline`

### 4.2 TG 查询与局部修复能力

`src/core/models/tg.py` 当前已支持：

- `find_schedulable_tasks()`
- `find_conflicting_tasks(task_id)`
- `find_tasks_blocked_by_gate(gate_id)`
- `find_tasks_requiring_resource(resource_key)`
- `find_retryable_tasks()`
- `mark_task_superseded(...)`
- `cancel_task(...)`
- `attach_outcome(...)`
- `replace_subgraph(...)`
- `collect_replan_frontier(...)`

### 4.3 Planner / Critic 职责

- Planner 输出结构化候选，不直接执行任务。
- TG Builder 只做结构转换，不做二次智能规划。
- Critic 识别重复任务、永久阻塞、失败饱和、低价值、失效任务，并给出局部替换或 replan 建议。

## 5. Runtime State 实现记录

### 5.1 核心模型

`src/core/models/runtime.py` 已定义：

- `RuntimeStatus`
- `TaskRuntimeStatus`
- `WorkerStatus`
- `SessionStatus`
- `LockStatus`
- `OperationRuntime`
- `TaskRuntime`
- `WorkerRuntime`
- `SessionRuntime`
- `ResourceLock`
- `BudgetRuntime`
- `CheckpointRuntime`
- `OutcomeCacheEntry`
- `RuntimeEventRef`
- `ReplanRequest`
- `RuntimeState`

### 5.2 Runtime 事件与 reducer

已实现文件：

- `src/core/runtime/events.py`
- `src/core/runtime/reducer.py`

已覆盖事件：

- `TaskQueued`
- `TaskStarted`
- `TaskCompleted`
- `TaskFailed`
- `TaskCancelled`
- `WorkerAssigned`
- `WorkerReleased`
- `LockAcquired`
- `LockReleased`
- `SessionOpened`
- `SessionExpired`
- `SessionHeartbeat`
- `BudgetConsumed`
- `CheckpointCreated`
- `ReplanRequested`

### 5.3 Runtime 存储与管理器

已实现文件：

- `src/core/runtime/store.py`
- `src/core/runtime/locks.py`
- `src/core/runtime/session_manager.py`
- `src/core/runtime/budgets.py`
- `src/core/runtime/checkpoint_store.py`
- `src/core/runtime/runtime_queries.py`
- `src/core/runtime/scheduler.py`

主要能力：

- `InMemoryRuntimeStore` 管理状态与事件日志。
- `RuntimeLockManager` 管理 TTL 锁与批量加锁。
- `RuntimeSessionManager` 管理 lease、heartbeat、复用与失效。
- `RuntimeBudgetManager` 管理 time / token / operation / noise / risk 预算。
- `RuntimeCheckpointManager` 管理 checkpoint、task lineage、replan marker、恢复上下文。
- `RuntimeQueryService` 面向 scheduler / critic 查询 active / queued / retryable / blocked 等运行时状态。
- `RuntimeScheduler` 基于 TG ready 状态和 Runtime 资源、预算、worker、session 做最小调度决策。

### 5.4 Runtime 与 TG / Planner / Critic 的兼容性

当前已满足：

- TG 只保留最小调度字段，不保存全部 runtime 细节。
- `TaskRuntime` 通过 `task_id` / `tg_node_id` 关联 TG。
- Critic 可从 Runtime State 读取连续失败、重试次数、预算耗尽、session 失效、resource lock 冲突。
- Scheduler 通过 `RuntimeQueryService` 判断 ready task 是否真可调度，以及哪些 task 被 runtime 阻塞。
- Checkpoint 与 replan marker 面向局部子图替换，不默认全图重建。
- 运行时事件不写入 KG / AG / TG。

### 5.5 当前命名兼容点

仍待统一的两个命名点：

- `RuntimeState.locks` 与需求文档中的 `resource_locks`
- `SessionRuntime.reusability` 与 `SessionReusePolicy` 的命名映射

当前 manager / query 层已做兼容处理。

## 6. Runtime 测试状态

已补测试文件：

- `tests/test_runtime_state.py`
- `tests/test_runtime_events.py`
- `tests/test_runtime_store.py`
- `tests/test_runtime_locks.py`
- `tests/test_runtime_sessions.py`
- `tests/test_runtime_budgets.py`
- `tests/test_runtime_checkpoint_store.py`
- `tests/test_runtime_scheduler.py`

另外为修复全量测试通过，还调整了：

- `src/core/planner/critic.py`
  兼容 `graph:ref_type:ref_id` 形式的 invalidation key。
- `src/core/graph/tg_builder.py`
  使用稳定时间戳，保证稳定图构建测试通过。
- `tests/test_kg_graph.py`
  放宽过严断言，改为检查关键实体包含关系。

当前测试结果：

- Runtime 相关测试通过。
- 全量测试通过：`73 passed`

## 7. Agent 层实现记录

本轮新增了独立 agent 协议层，不把所有逻辑继续堆进 orchestrator。

### 7.1 `src/core/agents/agent_protocol.py`

已定义：

- `AgentKind`
- `GraphScope`
- `WritePermission`
- `GraphRef`
- `AgentContext`
- `AgentInput`
- `AgentOutput`
- `AgentExecutionResult`
- `BaseAgent`

约定：

- 所有 agent 共用统一输入输出协议。
- `BaseAgent.run()` 是标准入口，负责输入校验、执行包装、最小权限检查、统一结果返回。
- 对 `state_deltas` 和 `emitted_events` 做最小权限约束。

### 7.2 `src/core/agents/agent_models.py`

已定义 agent 层共享交换模型：

- `ObservationRecord`
- `EvidenceRecord`
- `OutcomeRecord`
- `DecisionRecord`
- `StateDeltaRecord`
- `ReplanRequestRecord`

这些模型不依赖具体图实现，只作为 agent 层统一交换结构使用，并提供 `to_agent_output_fragment()`。

### 7.3 `src/core/agents/registry.py`

已定义：

- `AgentRegistry`
- `AgentRegistrationError`
- `AgentNotFoundError`

支持：

- `register(agent)`
- `unregister(agent_name)`
- `get(agent_name)`
- `list_all()`
- `list_by_kind(kind)`
- `dispatch(agent_name, agent_input)`

### 7.4 `src/core/agents/perception.py`

已新增：

- `PerceptionAgent`

职责：

- 消费 worker 产出的 `OutcomeRecord`
- 消费 raw execution result 与 task / graph refs
- 输出 `ObservationRecord`、`EvidenceRecord` 与 logs
- 将执行输出归一化为“可进入 KG 的结构化观测”

实现约束：

- 不做全局规划
- 不直接写 KG / AG / TG
- 不发 `state_deltas`
- 可根据 `outcome_type` 做分支解析
- 对缺失字段做稳健处理

当前已实现的私有方法：

- `_extract_observations(...)`
- `_extract_evidence(...)`
- `_infer_confidence(...)`

补充说明：

- `PerceptionAgent` 不是 KG writer，只负责解释与结构化，不直接修改 KG
- 当 `raw_result` 缺失时，可退化为仅使用 `OutcomeRecord.payload` 产出 observation / evidence
- 当前会自动补齐 task 级 `GraphRef`，并合并 input / outcome / raw result 中可解析的 refs

### 7.5 `src/core/agents/kg_events.py`

已新增：

- `KGDeltaEventType`
- `KGDeltaEvent`
- `KGEventBatch`

职责：

- 定义由 State Writer 发出的 KG 变化事件
- 作为 State Writer -> Graph Projection 的结构化交换层
- 提供批量封装方法，不绑定具体 store

当前事件类型包括：

- `entity_added`
- `entity_updated`
- `relation_added`
- `relation_updated`
- `confidence_changed`
- `state_invalidated`

### 7.6 `src/core/agents/state_writer.py`

已新增：

- `StateWriterAgent`
- `KGEntityPatch`
- `KGRelationPatch`
- `KGPatchApplyRequest`
- 临时本地 `KGDeltaEvent` 定义

职责：

- 消费 `ObservationRecord` / `EvidenceRecord`
- 将 observation / evidence 标准化为 KG patch / delta
- 输出 `state_deltas`、`emitted_events` 与 logs
- 留出给外部 KG store 应用 patch 的接口

实现约束：

- 只声明并写入 `KG` scope
- 不直接改 `TG`
- 不直接触发重规划
- 不把写库逻辑写死在 agent 内

当前已实现的私有方法包括：

- `_normalize_observation_to_entity_patch(...)`
- `_normalize_evidence_to_entity_patch(...)`
- `_normalize_evidence_to_relation_patch(...)`
- `_build_kg_delta_events(...)`

补充说明：

- 当前 `StateWriterAgent` 仍是 patch producer，不直接调用 `KnowledgeGraph`
- 文件内仍保留一份本地 `KGDeltaEvent`，后续可收敛到 `src/core/agents/kg_events.py`

### 7.7 `src/core/agents/graph_projection.py`

已新增：

- `GraphProjectionAgent`
- `AGProjectionEvent`

职责：

- 消费 `KGDeltaEvent` / `KGEventBatch`
- 将 KG 事实变化增量投影为 AG `StateNode` patch
- 实例化候选 `ActionNode` patch
- 输出 AG `state_deltas`、events 与 logs

实现约束：

- 只声明并写入 `AG` scope
- 默认走局部更新，不做全量 AG rebuild
- 不写 `TG`
- 为现有 `AttackGraphProjector` 预留兼容扩展点

当前已实现的私有方法包括：

- `_project_entity_to_state_nodes(...)`
- `_project_relation_to_state_nodes(...)`
- `_instantiate_action_candidates(...)`
- `_build_ag_delta(...)`

补充说明：

- 当前实现是“事件驱动 patch producer”，不直接应用到 `AttackGraph`
- `project_with_projector(...)` 提供了与 `AttackGraphProjector.project_incremental(...)` 对接的 handoff 形状

### 7.8 `src/core/agents/planner.py`

已新增：

- `PlannerAgent`
- `PlanningContext`
- `PlanningCandidate`

职责：

- 读取 AG 和 goal refs
- 在预算 / policy / runtime summary / critic hints 约束下搜索候选路径
- 对候选状态转移与动作链打分
- 选择 top-k 动作链并输出 `DecisionRecord`

实现约束：

- 不直接写 `KG`
- 不直接写 `TG`
- 不直接派发任务或执行工具
- decision 必须包含 `rationale`、`score`、`target_refs`

当前已实现的私有方法包括：

- `_collect_candidate_paths(...)`
- `_score_candidate(...)`
- `_select_top_k(...)`
- `_emit_decisions(...)`

补充说明：

- `PlanningCandidate` 当前直接复用现有 `TaskCandidate` 作为兼容候选类型
- `PlannerAgent` 已留出与 `src/core/planner/planner.py`、`src/core/planner/scorer.py` 对接的接口

### 7.9 `src/core/agents/task_builder.py`

已新增：

- `TaskBuilderAgent`
- `TaskBuildRequest`
- `TaskBuildResult`
- `TGDeltaEvent`

职责：

- 将 planner 的 decision / candidate actions 转换为 TG 子图 patch
- 生成任务节点、依赖、冲突、替代关系
- 生成任务组与 checkpoint 挂点
- 输出 TG `state_deltas`、events、可选 decisions 与 logs

实现约束：

- 不负责真实派发
- 不直接执行 worker
- 不写 `KG`
- 输出的是 TG patch / delta，供外部 TG store 应用

当前已实现的私有方法包括：

- `_build_task_nodes(...)`
- `_build_dependency_edges(...)`
- `_build_conflict_edges(...)`
- `_build_alternative_edges(...)`
- `_validate_tg_patch(...)`

补充说明：

- 当前优先复用现有 `TaskGraphBuilder.create_task_node(...)` 和 `TaskCandidate`
- 当输入里只有 action IDs 时，会退化生成最小可用的 skeletal `TaskCandidate`

### 7.10 Agent 层设计边界

当前约定：

- perception / planner / critic / worker / scheduler 等 agent 都通过统一协议通信。
- agent 不允许随意返回未定义结构。
- worker 不直接生成 KG 事实节点，不直接写 AG / TG 主结构。
- perception 不直接生成 KG 事实写入，只生成可供 state writer 消费的 observation / evidence。
- graph projection 和 state writer 仍保留为专属 owner。
- planner 只输出 decision / planning candidate，不直接落 TG。
- task builder 只把 planner 结果转成 TG patch，不直接派发 worker。

### 7.11 `src/core/agents/agent_pipeline.py`

已新增：

- `PipelineStepResult`
- `PipelineCycleResult`
- `AgentPipeline`

职责：

- 提供最小可行 agent 编排层
- 串联：
  - `run_planning_cycle(...)`
  - `run_execution_cycle(...)`
  - `run_feedback_cycle(...)`
- 支持按 `agent name` 或 `agent kind` 解析 agent
- 每一步统一走 `AgentInput -> AgentOutput`
- 负责 step 聚合、cycle 聚合和可选 sink 转发

实现约束：

- pipeline 不接管 planner / scheduler / critic / worker 的业务判断
- pipeline 不直接写 store
- pipeline 不直接重规划
- pipeline 只负责 envelope 组装、调用顺序和结果转发

当前已实现的最小兼容能力：

- `event_sink` / `state_delta_sink`
- `StateWriter state_deltas -> KGEventBatch` 的轻量适配
- worker handoff 时的跨层 `GraphRef` 归一化
- `Critic.recent_outcomes` 所需 runtime 风格结果归一化

补充说明：

- 当前 `AgentPipeline` 已可跑通最小单轮闭环，但仍属于轻量 orchestrator
- 未来若接 runtime store / event bus，应优先复用 sink 和 cycle 入口，而不是把业务逻辑回灌到 pipeline

### 7.12 当前 agent 框架分析

从现有代码看，agent 层已经形成较清晰的三段式框架：

1. 协议层：

- `agent_protocol.py`
- `agent_models.py`
- `registry.py`

职责是统一 envelope、权限声明、结构化 records 和 dispatch 入口。

2. owner agent 层：

- `PerceptionAgent`
- `StateWriterAgent`
- `GraphProjectionAgent`
- `PlannerAgent`
- `TaskBuilderAgent`
- `SchedulerAgent`
- `CriticAgent`

这些 agent 都是明确带 owner 边界的组件：

- `StateWriter` 拥有 KG patch 生产
- `GraphProjection` 拥有 AG patch 生产
- `TaskBuilder / Scheduler` 拥有 TG / Runtime 侧 patch 与 assignment 决策
- `Critic` 只拥有 advisory / replan recommendation

3. worker 执行层：

- `BaseWorkerAgent`
- `ReconWorker`
- `AccessValidationWorker`
- `PrivilegeValidationWorker`
- `GoalValidationWorker`

worker 只做 task execution 和 outcome/raw result 返回，不拥有图写入权。

当前框架的核心优点：

- owner 边界明确，避免 KG / AG / TG / Runtime 互相串写
- patch / delta 优先，便于后续接 store reducer / event bus
- `AgentRegistry + AgentPipeline` 已足够支撑最小闭环
- 测试已开始覆盖权限边界和协议边界，而不仅是 happy path

当前框架的主要不足：

- `StateWriter` 仍保留本地 `KGDeltaEvent`，尚未完全收敛到 `kg_events.py`
- agent protocol `GraphRef` 与 AG/TG/KG model 内部 `GraphRef` 仍存在跨层适配成本
- pipeline 目前是轻量编排器，还没有正式接到统一 runtime store / event log / apply path
- 现有 orchestrator / app 层还未把这套 agent pipeline 作为主执行路径

## 8. Worker 层实现记录

### 8.1 `src/core/workers/base.py`

当前文件包含两层：

1. 新的 agent 化 worker 基类：

- `WorkerCapability`
- `WorkerTaskSpec`
- `BaseWorkerAgent`

特性：

- `kind` 固定为 `worker`
- 默认仅声明 `RUNTIME` scope，且无 KG / AG / TG structural write 权限
- 提供：
  - `supports_task(task_spec)`
  - `execute_task(task_spec, agent_input)`
  - `build_outcome(...)`
  - `build_raw_result(...)`
- worker 输出必须通过 `OutcomeRecord`
- worker 不得直接对 KG / AG / TG 发 `state_deltas`

2. 兼容层：

- `BaseWorker`
- `WorkerExecutionError`
- `WorkerBlockedError`

兼容层用于保持现有 `goal/recon/access` worker 不被这次重构打断。

### 8.2 现有兼容 worker

已实现文件：

- `src/core/workers/recon_worker.py`
- `src/core/workers/access_worker.py`
- `src/core/workers/goal_worker.py`
- `src/core/workers/access_validation_worker.py`
- `src/core/workers/privilege_validation_worker.py`
- `src/core/workers/goal_validation_worker.py`

它们当前仍走旧的 worker 请求/结果协议，但内部边界已遵守：

- 只返回 observation / evidence / write intent / projection request / runtime request / replan hint / critic signal
- 不直接改 KG / AG / TG 主结构

### 8.3 `src/core/workers/general_worker.py`

新增 MVP 通用 worker：

- `GeneralWorkerAgent`

支持任务类型：

- `recon`
- `access_validation`
- `goal_validation`

约定：

- 输入来自 `WorkerTaskSpec`
- 当前执行逻辑是 placeholder / fake executor
- 输出一个 `OutcomeRecord`
- 输出一个 raw result 结构，放入 `AgentOutput.evidence`
- 不调用真实攻击工具
- 为未来真实工具接入预留扩展点：
  - `_execute_recon(...)`
  - `_execute_access_validation(...)`
  - `_execute_goal_validation(...)`

## 9. Agent / Worker 测试状态

已新增：

- `tests/test_agent_workers.py`
- `tests/test_agents.py`

当前覆盖：

- worker request 构建
- recon worker 结构化输出
- access worker 的 session 阻塞路径
- access worker 成功路径与 critic / replan 信号
- goal worker 的 local replan 路径
- 错误 agent role 的输入校验
- worker 不能直接写 KG / AG / TG structural state
- `StateWriter` 只能写 KG
- `GraphProjection` 只能写 AG
- `TaskBuilder` 只能写 TG
- `Planner` 只输出 decision，不直接执行
- `Critic` 可产出 replan request，且不污染 KG
- `Perception` 可将 outcome 转 observation / evidence
- `Scheduler` 能对 ready task 产出 assignment 决策
- `AgentRegistry` 能正常 dispatch
- `AgentPipeline` 能跑通最小单轮 cycle

当前测试结果：

- `pytest tests/test_agents.py -q` 通过
- agent 层最小闭环测试已覆盖 planning / execution / feedback 三段

本轮测试驱动下顺手修复的协议兼容问题：

- `TaskBuilderAgent` 的 checkpoint `anchor_refs` 改为使用 TG/AG model 侧 `GraphRef`
- `GraphProjectionAgent` 的 `ProjectionTrace.input_refs` 改为使用 AG model 侧 `GraphRef`
- `AgentPipeline` 新增：
  - 跨层 `GraphRef` 归一化
  - `query -> ag` 的兼容映射
  - `Critic recent_outcomes` 的 runtime 风格归一化

当前缺口：

- 目前仍缺少更细粒度的 negative-path 测试
- 目前还没有针对外部 store apply path / event bus 的集成测试
- 目前还没有针对多 worker / 多 task / 多轮 cycle 的组合测试

## 10. 后续建议顺序

1. 为 `PerceptionAgent` 补更细粒度的 negative-path 测试
2. 为 `AgentRegistry` 补更多错误路径与重复注册测试
3. 为 `GeneralWorkerAgent` 补 `tests/test_general_worker.py`
4. 为 `AgentPipeline` 增加多轮 cycle、异常恢复、event sink / delta sink 的集成测试
5. 将新的 `AgentPipeline` 接到 orchestrator / dispatcher，替代散落的手工串联
6. 将 `state_writer` / `graph_projection` / `planner` / `task_builder` 接到正式 store apply / event bus 路径
7. 统一 `src/core/agents/state_writer.py` 中本地 `KGDeltaEvent` 与 `src/core/agents/kg_events.py`
8. 逐步让现有 `recon_worker` / `access_worker` / `goal_worker` 迁移到 `BaseWorkerAgent`
9. 统一 `RuntimeState.locks` 与 `resource_locks` 命名
10. 统一 `SessionRuntime.reusability` 与 `SessionReusePolicy` 命名
11. 后续记忆文档只通过 `apply_patch` 维护，避免再次出现编码损坏

## 11. 历史恢复补录

本节用于补回此前因文档乱码而丢失、但已在对话中确认过或可从当前代码中重新核实的内容。这里不是逐字恢复旧文，而是基于已确认信息做结构化恢复。

### 11.1 现有代码框架总览

系统主链路可概括为：

- `Knowledge Graph (KG) -> Attack Graph (AG) -> Planner -> Task Graph (TG) -> Runtime / Worker / Agent`

各层含义：

- KG 是事实层，承载主机、服务、身份、凭据、权限、会话、数据资产、目标、证据等长期世界事实。
- AG 是决策层，将世界事实投影为可规划状态、动作、目标和约束。
- Planner 是决策候选层，做启发式搜索和候选链路选择，不直接执行。
- TG 是执行结构层，将 AG 动作收敛为任务节点、依赖、冲突、替代、恢复锚点和结果挂接关系。
- Runtime 是执行控制层，维护任务运行态、worker、session、锁、预算、checkpoint、事件、replan 请求。
- Agent / Worker 是执行与协议层，所有组件通过统一输入输出协议通信，不直接跨层写主结构。

### 11.2 现有实现成熟度

当前已较完整落地的部分：

- KG 模型与图容器
- AG 模型与 KG -> AG 投影
- Planner / Scorer / Critic
- TG 模型与 AG -> TG 构建
- Runtime State 模型、事件、reducer、store、locks、sessions、budgets、checkpoint、queries、scheduler
- Agent 协议层、共享模型、registry、worker 基类和 MVP general worker

仍属于骨架或后续扩展点的部分：

- `src/app/orchestrator.py`
- `src/app/api.py`
- 正式版 orchestrator / dispatcher / worker runner
- General worker 对真实执行器的接入

当前已新增但仍偏 patch producer / integration skeleton 的部分：

- `src/core/agents/kg_events.py`
- `src/core/agents/state_writer.py`
- `src/core/agents/graph_projection.py`
- `src/core/agents/planner.py`
- `src/core/agents/task_builder.py`
- `src/core/agents/scheduler_agent.py`
- `src/core/agents/critic.py`
- `src/core/agents/agent_pipeline.py`

这些模块当前已具备协议层输入输出与 owner 边界，但尚未真正接到统一 orchestrator / store apply path。

### 11.3 当前测试重点

从当前测试布局可以反推出仓库当前设计重心：

- 测试重点已经覆盖 KG / AG / Planner / TG / Critic / Runtime / Agent 协议
- 当前重心仍然是“图模型 + 规划机制 + 运行时控制面”
- 真正的工具执行、网络联动、攻击面交互仍未纳入实现重心

### 11.4 Agent 层与旧设计的关系

新增 agent 层后，职责关系进一步明确为：

- perception：负责感知与观察，不直接写事实图
- perception：可消费 `OutcomeRecord + raw result + refs`，产出 `ObservationRecord / EvidenceRecord / logs`
- state_writer：唯一正式落 KG 的写入方
- graph_projection：唯一正式更新 AG 的投影方
- planner：只产出 decision / candidate / request
- task_builder：将候选转成 TG 结构
- scheduler：基于 TG + Runtime 做最小调度决策
- agent_pipeline：只串联 cycle，不承接 owner 逻辑
- worker：执行任务并返回 outcome / raw result / runtime request / hint
- critic：只输出 cancel / replace / replan 建议

### 11.5 Worker 层恢复说明

此前对话里已确认的 worker 设计边界如下，现补回文档：

- Worker 不是 KG writer，不直接构造 KG 事实节点
- Worker 不是 AG updater，不直接构造或变更 AG 主结构
- Worker 不是 TG owner，不直接增删改 TG 主结构
- Worker 只执行任务，并通过统一协议产出：
  - `OutcomeRecord`
  - raw result 引用
  - evidence / observation
  - runtime request
  - replan hint / critic signal

### 11.6 当前已确认的代码状态

当前仓库中与 agent / worker 新增实现直接相关的文件包括：

- `src/core/agents/agent_protocol.py`
- `src/core/agents/agent_models.py`
- `src/core/agents/kg_events.py`
- `src/core/agents/perception.py`
- `src/core/agents/registry.py`
- `src/core/agents/state_writer.py`
- `src/core/agents/graph_projection.py`
- `src/core/agents/planner.py`
- `src/core/agents/task_builder.py`
- `src/core/agents/scheduler_agent.py`
- `src/core/agents/critic.py`
- `src/core/agents/agent_pipeline.py`
- `src/core/workers/base.py`
- `src/core/workers/general_worker.py`
- `src/core/workers/access_validation_worker.py`
- `src/core/workers/privilege_validation_worker.py`
- `src/core/workers/goal_validation_worker.py`
- `tests/test_agent_workers.py`
- `tests/test_agents.py`

当前已确认测试状态：

- 新增 agent / worker 相关测试通过
- `tests/test_agents.py` 通过：`10 passed`
- 之前记录的全量通过值 `73 passed` 已过时，应以下一次全量回归结果为准

### 11.7 恢复说明

本节内容的恢复来源只有两类：

- 你此前在对话中明确确认过的架构与实现要求
- 当前工作区代码中可以直接核实的实现状态

由于当前目录不是 git 仓库，且不存在旧版记忆文档备份，所以无法对原乱码部分做逐字恢复；本节是“基于已知事实的补录恢复”。

## 12. 2026-04-15 第一阶段控制面改造记录

本轮修改目标是把当前仓库从“纯内核/骨架”推进到“第一阶段可启动控制面”状态，重点只覆盖：

- 应用配置系统
- 可持久化 Runtime Store
- 最小应用服务层
- 可启动 FastAPI 入口

### 12.1 新增 / 修改文件

- `src/app/settings.py`
- `src/app/orchestrator.py`
- `src/app/api.py`
- `src/core/runtime/store.py`
- `requirements.txt`
- `FASTAPI_SETUP.md`
- `tests/test_app_orchestrator.py`
- `tests/test_runtime_store.py`

### 12.2 第一阶段应用层职责

当前 `src/app` 已不再是纯占位：

- `settings.py`
  提供 `AppSettings`，从 `AEGRA_*` 环境变量读取运行配置。
- `orchestrator.py`
  提供 `AppOrchestrator`、`TargetHost`、`OperationSummary`。
- `api.py`
  提供 FastAPI 控制面工厂 `create_app()` 和全局 `app`。

### 12.3 已落地的第一阶段能力

`AppOrchestrator` 当前已支持：

- `create_operation(operation_id, metadata)`
- `import_targets(operation_id, targets)`
- `start_operation(operation_id)`
- `get_operation_state(operation_id)`
- `get_operation_summary(operation_id)`
- `list_operations()`

当前第一阶段目标导入策略：

- 目标 inventory 保存在 `RuntimeState.execution.metadata`
- key 为：
  - `target_inventory`
  - `target_count`
  - `control_plane`
  - `last_control_cycle`

说明：

- 这仍属于“控制面 bootstrap”而不是正式扫描执行面。
- 目标还没有进入 KG / AG / TG 正式事实流。
- 当前设计是先让控制面与运行时持久化稳定，再进入第二阶段 worker/tool 接入。

### 12.4 Runtime Store 扩展

`src/core/runtime/store.py` 在原有 `InMemoryRuntimeStore` 之外新增：

- `FileRuntimeStore`

其能力：

- 使用 JSON 文件持久化 `RuntimeState`
- 使用 JSON 文件持久化 append-only 事件日志
- 支持：
  - `create_operation`
  - `save_state`
  - `get_state`
  - `append_event`
  - `list_events`
  - `apply_event`
  - `snapshot`
  - `delete_operation`
  - `list_operation_ids`

同时 `RuntimeStore` 抽象接口新增：

- `list_operation_ids()`

当前定位：

- `FileRuntimeStore` 是第一阶段本地持久化方案
- 不是 Redis / Postgres 级别的正式分布式存储
- 适合本地调试、单机场景、控制面验证、测试环境落地

### 12.5 FastAPI 控制面状态

当前 `src/app/api.py` 已具备最小 HTTP 控制面：

- `GET /`
- `GET /health`
- `GET /operations`
- `POST /operations`
- `POST /operations/{operation_id}/targets`
- `POST /operations/{operation_id}/start`
- `GET /operations/{operation_id}`
- `GET /operations/{operation_id}/summary`

并且：

- 已暴露全局 `app`
- 安装依赖后可直接用 `uvicorn src.app.api:app` 启动
- `/docs` 和 `/openapi.json` 可用

兼容处理：

- 当前工作区默认未安装 `fastapi` / `uvicorn`
- 因此 `api.py` 保留了“未安装时给出明确错误”的逻辑
- 在未安装依赖的环境下，模块仍可安全导入，`app` 为 `None`

### 12.6 依赖与启动说明

新增文件：

- `requirements.txt`
- `FASTAPI_SETUP.md`

当前推荐安装命令：

```powershell
python -m pip install -r requirements.txt
```

当前推荐启动命令：

```powershell
python -m uvicorn src.app.api:app --host 127.0.0.1 --port 8000 --reload
```

启动后关键地址：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

### 12.7 第一阶段配置项

`AppSettings` 当前支持的主要环境变量包括：

- `AEGRA_RUNTIME_STORE_BACKEND`
- `AEGRA_RUNTIME_STORE_DIR`
- `AEGRA_CONTROL_API_TITLE`
- `AEGRA_CONTROL_API_VERSION`
- `AEGRA_MAX_CONCURRENT_WORKERS`
- `AEGRA_DEFAULT_OPERATION_BUDGET`
- `AEGRA_DEFAULT_SCAN_TIMEOUT_SEC`
- `AEGRA_AUDIT_ENABLED`
- `AEGRA_LLM_API_KEY`
- `AEGRA_LLM_BASE_URL`
- `AEGRA_TOOL_NMAP_PATH`
- `AEGRA_TOOL_PYTHON_PATH`

说明：

- 其中 `LLM API` 和 `tool path` 目前只是配置占位
- 当前第一阶段尚未真正接入 LLM 和真实扫描器执行逻辑

### 12.8 设计结论更新

关于 `FastAPI` 的当前结论：

- 当前系统更适合 `FastAPI` 而不是 `Django`
- 原因是项目主体是：
  - `Pydantic` 模型
  - Runtime / Store / Orchestrator 控制面
  - 无传统 ORM 后台和模板页面需求
- 因此 `FastAPI` 更适合作为：
  - 控制 API 层
  - 调度入口层
  - 前后端分离的服务端控制面

当前架构建议仍为：

- `src/core/*` 保持纯引擎
- `src/app/orchestrator.py` 作为应用服务层
- `FastAPI` 只作为 HTTP 控制壳层

### 12.9 当前限制

尽管第一阶段已完成，当前仓库依然不等于“多主机渗透平台成品”。

仍未落地的关键部分包括：

- 真实工具执行器接入
- Recon / Access / Goal worker 的真实外部能力
- 目标导入到 KG 的正式事实写入流
- Redis / Postgres 级别持久化
- Session / Credential / Pivot 的真实执行控制
- 审批、授权范围控制、审计链路完善

因此当前状态更准确描述为：

- 已有可启动控制面
- 已有本地可持久化 runtime store
- 已有 operation / target inventory / bootstrap start 管理能力
- 尚未进入真实多主机扫描执行阶段

### 12.10 最新测试状态

本轮新增或更新的测试包括：

- `tests/test_app_orchestrator.py`
- `tests/test_runtime_store.py`

当前已确认：

- 第一阶段新增测试通过
- 全量测试通过：`88 passed`

该数值替代此前文档中已经过时的：

- `73 passed`
- `83 passed`
- `87 passed`

后续如再有全量回归，应以最近一次实际执行结果覆盖本节。

## 13. 2026-04-15 第二阶段执行器与运行时扩展记录

本轮修改目标是把第二阶段的 worker 从“placeholder / metadata 驱动”推进到“可消费真实执行结果与运行时状态”的形态，重点覆盖：

- 统一 ToolRunner 层
- ReconWorker 真正执行外部命令
- AccessWorker / GoalWorker 优先消费 runtime-backed 视图
- Runtime 新增 credential / session lease / pivot route 结构

### 13.1 本轮新增 / 修改文件

新增文件：

- `src/core/workers/tool_runner.py`
- `tests/test_phase_two_workers.py`

修改文件：

- `src/core/workers/recon_worker.py`
- `src/core/workers/access_worker.py`
- `src/core/workers/goal_worker.py`
- `src/core/models/runtime.py`

### 13.2 ToolRunner 层

新增 `src/core/workers/tool_runner.py`，提供统一外部命令执行抽象：

- `ToolExecutionSpec`
- `ToolExecutionResult`
- `ToolRunner`

当前已覆盖能力：

- 命令执行
- 超时控制
- stdout / stderr 捕获
- retries
- 退出码判定
- 错误归类：
  - `success`
  - `nonzero_exit`
  - `timeout`
  - `command_not_found`
  - `process_error`

设计边界：

- ToolRunner 只负责进程执行与结果归一化
- 不负责业务解析
- Worker 负责把 stdout / stderr 解释为业务语义并转成 observation / evidence / fact write / runtime request

### 13.3 ReconWorker 第二阶段状态

`src/core/workers/recon_worker.py` 已从 placeholder adapter 改为真正的执行器适配层。

当前能力：

- 通过 `ToolRunner` 执行外部命令
- 支持从 `metadata["tool_command"]` 注入明确命令
- 支持：
  - host discovery
  - service validation
  - identity/context discovery
- 对工具输出做统一解析：
  - JSON stdout 优先解析
  - 非 JSON stdout 退化为 raw text payload

当前 ReconWorker 输出新增了更明确的结构：

- `tool`
  包含规范化后的命令执行结果
- `parsed`
  包含从 stdout 提取的结构化 payload

兼容性说明：

- 仍然保留旧版 compatibility worker request/result 路径
- 因此原有 `tests/test_agent_workers.py` 不需要推翻
- 同时新增第二阶段测试覆盖真实命令执行路径

当前默认命令策略：

- 若 `tool_command` 未提供，则生成一个最小的 Python 子进程探测命令
- 该默认行为仍属于“本地模拟执行器”
- 它的意义是打通执行链路、超时/错误归一化和输出标准化
- 真实环境下应由后续接入实际扫描器命令替换

### 13.4 AccessWorker 第二阶段状态

`src/core/workers/access_worker.py` 当前已不再仅依赖布尔 metadata 假设，而是优先读取运行时上下文视图：

- `runtime_session`
- `runtime_credential`
- `host_reachability`
- `privilege_validation`

当前行为调整：

- 若 host 不可达，则返回 `BLOCKED`
- 若需要 session 且 session 不可用，则返回 `OPEN_SESSION` runtime request
- 若需要 credential 且 credential 非 `valid`，则返回 `FAILED`
- 若 privilege validation 未通过，则仍可输出成功的 access validation 结果，同时发出：
  - `critic_signal`
  - `replan_hint`

当前 AccessWorker 的 `outcome_payload` 已新增：

- `credential_status`
- `reachable`
- `privilege_validation`

### 13.5 GoalWorker 第二阶段状态

`src/core/workers/goal_worker.py` 当前优先读取结构化：

- `goal_evaluation`

而不再只依赖：

- `goal_satisfied`

当前新增逻辑：

- `goal_evaluation.satisfied`
- `goal_evaluation.missing_requirements`
- `goal_evaluation.validated_ref_ids`

输出变化：

- observation payload 中包含 `goal_evaluation`
- fact write 中包含 `missing_requirements`
- `outcome_payload` 中包含完整 `goal_evaluation`

这使 GoalWorker 可以直接消费后续真实验证器返回的结构化目标判断结果。

### 13.6 Runtime 模型第二阶段扩展

`src/core/models/runtime.py` 本轮新增：

- `CredentialStatus`
- `CredentialKind`
- `PivotRouteStatus`
- `CredentialRuntime`
- `SessionLeaseRuntime`
- `PivotRouteRuntime`

并对 `RuntimeState` 增加：

- `credentials`
- `session_leases`
- `pivot_routes`

同时新增方法：

- `add_credential(...)`
- `add_session_lease(...)`
- `add_pivot_route(...)`

#### 13.6.1 CredentialRuntime

当前字段包括：

- `credential_id`
- `kind`
- `principal`
- `secret_ref`
- `status`
- `bound_targets`
- `source_session_id`
- `last_validated_at`
- `failure_count`
- `metadata`

已提供：

- `is_usable()`

#### 13.6.2 SessionLeaseRuntime

当前字段包括：

- `lease_id`
- `session_id`
- `owner_task_id`
- `owner_worker_id`
- `acquired_at`
- `lease_expiry`
- `reuse_policy`
- `metadata`

已提供：

- `is_active()`

#### 13.6.3 PivotRouteRuntime

当前字段包括：

- `route_id`
- `destination_host`
- `source_host`
- `via_host`
- `session_id`
- `status`
- `protocol`
- `last_verified_at`
- `metadata`

已提供：

- `is_usable()`

### 13.7 当前第二阶段定位

本轮并没有完成“真实多主机渗透平台”，但已经补齐了第二阶段中最关键的执行骨架：

- Worker 已能真正执行命令
- Worker 已能区分执行成功 / 失败 / 超时 / 找不到命令
- Runtime 已能表达：
  - 凭据
  - 会话租约
  - pivot 路径
- Access / Goal worker 已能消费真实 runtime 视图而不是只靠 mock bool

因此当前状态应表述为：

- 已具备“真实执行器接入前的标准命令执行抽象”
- 已具备“运行时会话 / 凭据 / pivot 结构”
- 已具备“可向真实扫描器与真实校验器过渡的 worker 形态”

### 13.8 2026-04-16 第二阶段补充修改：结果消费层与真实探测接入

在 2026-04-15 第二阶段骨架之上，又补充了两类关键能力：

- phase-two 统一结果消费层
- ReconWorker 真实探测适配层

本次补充后，第二阶段主链可更准确概括为：

- `真实工具/验证器输出 -> Worker 标准化 -> AgentTaskResult -> Runtime / StateWriter / GraphProjection -> KG / AG`

#### 13.8.1 phase-two 统一结果消费层

新增文件：

- `src/core/runtime/result_applier.py`

新增：

- `PhaseTwoApplyResult`
- `PhaseTwoResultApplier`

职责：

- 消费 `AgentTaskResult`
- 调用现有 runtime managers 处理 `runtime_requests`
- 将 `checkpoint_hints / replan_hints / critic_signals` 正式写入 runtime state / runtime event queue
- 将 observation / evidence 交给 `StateWriterAgent`
- 将 `fact_write_requests` 转为正式 KG structural deltas
- 将 `projection_requests` 交给 `GraphProjectionAgent`

设计边界：

- worker 仍然不能直接写 KG / AG / TG 主结构
- `StateWriterAgent` 仍是正式 KG owner
- `GraphProjectionAgent` 仍是正式 AG owner
- result applier 只是 phase-two 的统一消费与路由层，不替代 owner agent

#### 13.8.2 ReconWorker 真实探测接入

新增文件：

- `src/core/workers/probe_adapters.py`

修改文件：

- `src/core/workers/recon_worker.py`

本次新增的探测适配器包括：

- `NmapAdapter`
- `MasscanAdapter`
- `CustomProbeAdapter`

当前 ReconWorker 的职责已收敛为：

- 根据 task type / metadata 选择 probe adapter
- 通过 `ToolRunner` 执行真实工具命令
- 解析原始输出
- 归一化为统一结构：
  - `entities`
  - `relations`
  - `evidence`
  - `runtime_hints`
  - `confidence`
  - `summary`
- 生成兼容现有协议的 `AgentTaskResult`

当前默认命令策略已不再使用伪造成功的 Python 模拟结果，而是：

- 优先选择实际 probe adapter
- 工具不可用时返回明确的 `blocked / failed` 语义

### 13.9 2026-04-16 第二阶段补充修改：KG 正式事实写入流

本次补充修改将第二阶段 worker 结果从“粗粒度 assertion”推进到“正式 KG fact write 流”。

修改文件：

- `src/core/workers/recon_worker.py`
- `src/core/workers/access_worker.py`
- `src/core/workers/goal_worker.py`

#### 13.9.1 ReconWorker 事实写入升级

`ReconWorker` 当前不再只写单条 assertion，而是会生成：

- `FactWriteKind.ENTITY_UPSERT`
- `FactWriteKind.RELATION_UPSERT`

当前统一写入的事实包括：

- `Host`
- `Service`
- `Identity`
- Reachability 状态
- `SUPPORTED_BY`
- `HOSTS`
- 其他从 probe result 中解析出的结构化 relation

#### 13.9.2 AccessWorker 事实写入升级

`AccessWorker` 当前会生成结构化 access 事实，而不再只是访问断言。

当前统一写入的事实包括：

- `Session`
- `Credential`
- `PrivilegeState`
- target 上的 access validation 状态
- `SESSION_ON`
- `AUTHENTICATES_AS`
- `CAN_REACH`
- `HAS_PRIVILEGE_STATE`
- 与 evidence 的 `SUPPORTED_BY`

#### 13.9.3 GoalWorker 事实写入升级

`GoalWorker` 当前会把目标验证结果写成结构化 goal 事实，而不再只写一条 goal assertion。

当前统一写入的事实包括：

- `Goal`
- `Finding`
- `goal_satisfied`
- `missing_requirements`
- `validated_ref_ids`
- `TARGETS`
- `RELATED_TO`
- 与 evidence 的 `SUPPORTED_BY`

#### 13.9.4 KG owner 边界保持不变

尽管 worker 现在会生成结构化 `fact_write_requests`，但当前 owner 边界没有变化：

- worker 只提出 `FactWriteRequest`
- `PhaseTwoResultApplier` 负责把 `fact_write_requests` 转为 KG structural state deltas
- `StateWriterAgent` 仍然是正式 KG owner

因此当前第二阶段已经完成了此前待办项中的：

- 真正接入 `nmap / masscan / 自定义探测器` 适配层
- 将 ReconWorker 的默认命令切换为真实工具优先
- 让工具结果正式进入 KG 的事实写入流

### 13.10 当前仍未完成的第二阶段内容

尽管第二阶段又向前推进了一步，以下内容仍未真正完成：

- Session lease 与 credential 的 manager 层实现
- Pivot route 的动态维护逻辑
- AccessWorker 对真实会话探测结果、凭据校验器输出、reachability source 的统一接入
- GoalWorker 对真实目标验证器输出的正式对接

所以当前第二阶段实现更准确应叫：

- `phase-two execution substrate`

而不是完整 finished execution platform。

### 13.11 2026-04-16 第二阶段补充修改：runtime manager 层

本次补充修改把第二阶段中原先只停留在 `RuntimeState` 数据模型里的 credential / lease / pivot route，推进到了正式 manager 层。

新增文件：

- `src/core/runtime/credential_manager.py`
- `src/core/runtime/lease_manager.py`
- `src/core/runtime/pivot_route_manager.py`

#### 13.11.1 RuntimeCredentialManager

新增：

- `RuntimeCredentialManager`

当前已提供：

- `upsert_credential(...)`
- `mark_valid(...)`
- `mark_invalid(...)`
- `mark_expired(...)`
- `mark_revoked(...)`
- `bind_target(...)`
- `record_validation(...)`

风格保持与 `RuntimeSessionManager` 一致：

- 直接操作 `RuntimeState`
- 目标不存在时抛 `ValueError`
- 状态变更时统一更新 `last_updated`
- 生命周期逻辑收敛在 manager 中，而不是散落在 worker 中

#### 13.11.2 RuntimeLeaseManager

新增：

- `RuntimeLeaseManager`

当前已提供：

- `create_lease(...)`
- `extend_lease(...)`
- `release_lease(...)`
- `cleanup_expired_leases(...)`
- `bind_lease_to_task_or_session(...)`

当前定位：

- lease manager 负责 task/session 绑定和 lease 生命周期
- 不把 lease 续期、释放、过期判定塞回 worker

#### 13.11.3 RuntimePivotRouteManager

新增：

- `RuntimePivotRouteManager`

当前已提供：

- `register_candidate(...)`
- `activate_route(...)`
- `fail_route(...)`
- `close_route(...)`
- `refresh_from_reachability(...)`
- `select_best_route(...)`

当前定位：

- pivot route manager 负责 route candidate / active / failed / closed 状态转换
- `refresh_from_reachability(...)` 负责把 reachability 信号映射为 route lifecycle
- `select_best_route(...)` 负责 destination host 的当前最优 route 选择

### 13.12 2026-04-16 第二阶段补充修改：AccessWorker runtime snapshot 接入

本次补充修改把 `AccessWorker` 从“读取 metadata 拼视图”推进为“消费标准化 execution context + runtime managers”的实现。

修改文件：

- `src/core/workers/access_worker.py`

#### 13.12.1 AccessExecutionContext

`AccessWorker` 内部新增：

- `AccessExecutionContext`

用于统一承载：

- `runtime_snapshot`
- `session_probe`
- `credential_validation`
- `reachability`
- `privilege_validation`
- `selected_session_id`
- `selected_route`
- `require_session`
- `require_credential`

这意味着：

- AccessWorker 不再在主流程里散读多处 metadata
- runtime snapshot 和 validator 输出先被归一化，再进入决策逻辑

#### 13.12.2 AccessWorker 当前第二阶段能力

当前 AccessWorker 已能统一接入：

- 真实会话探测结果
- 凭据校验器输出
- reachability source：
  - `direct`
  - `session`
  - `pivot`

并基于 runtime managers 做最小决策：

- 复用现有 session
- session 缺失时申请新 session
- 选择当前可用 pivot route
- 记录 credential validation 结果
- 在 privilege gap 场景下触发 critic / replan

当前实现说明：

- 没有扩张现有 worker 协议
- 仍然沿用当前 `AgentTaskRequest -> AgentTaskResult` 兼容路径
- route 选择结果通过 observation / fact write / projection metadata / runtime request metadata 向后传递

### 13.13 2026-04-16 第二阶段补充修改：GoalWorker validator 接入

本次补充修改把 `GoalWorker` 从“直接读取 metadata.goal_evaluation”推进为“消费标准 GoalValidator 输出”的实现。

新增文件：

- `src/core/workers/goal_validator.py`

修改文件：

- `src/core/workers/goal_worker.py`

#### 13.13.1 GoalValidator 抽象

新增：

- `GoalEvaluation`
- `GoalValidator`
- `MetadataGoalValidator`

当前 `GoalEvaluation` 至少统一包含：

- `satisfied`
- `missing_requirements`
- `validated_ref_ids`
- `supporting_evidence`
- `confidence`

当前 `MetadataGoalValidator` 的定位是：

- 作为第二阶段的兼容 validator
- 把现有 `goal_validator_output / goal_evaluation / goal_satisfied` 归一化为标准 `GoalEvaluation`
- 为后续真实目标验证器替换留出稳定接口

#### 13.13.2 GoalWorker 当前第二阶段能力

当前 GoalWorker 已改为：

- 调用 `GoalValidator.evaluate(...)`
- 消费标准 `GoalEvaluation`
- 基于验证结果生成：
  - `fact_write_requests`
  - `checkpoint_hints`
  - `replan_hints`
  - `critic_signals`
  - `runtime_requests`

当前行为：

- satisfied 路径：
  - 返回 `SUCCEEDED`
  - 生成 checkpoint hint
- unsatisfied 路径：
  - 返回 `NEEDS_REPLAN`
  - 明确生成 `REQUEST_REPLAN`
  - 明确生成 critic signal 和 replan hint

因此 GoalWorker 现已具备正式对接真实目标验证器输出的协议入口。

### 13.14 当前仍未完成的第二阶段内容

尽管第二阶段又向前推进了一步，以下内容仍未真正完成：

- Pivot route 的动态维护逻辑仍主要停留在 manager 能力层，尚未完全接入统一 runtime orchestration
- AccessWorker 当前仍是“runtime snapshot + validator output”驱动，尚未直接对接独立 access validator 组件或真实会话探测器实现
- GoalWorker 当前默认 validator 仍是 `MetadataGoalValidator`，尚未接入独立真实目标验证器实现

所以当前第二阶段实现更准确应叫：

- `phase-two execution substrate`

而不是完整 finished execution platform。

### 13.15 第二阶段测试状态

截至 2026-04-16，本轮第二阶段新增或更新的测试文件包括：

- `tests/test_phase_two_workers.py`
- `tests/test_phase_two_result_applier.py`
- `tests/test_agent_workers.py`
- `tests/test_runtime_credentials.py`
- `tests/test_runtime_leases.py`
- `tests/test_runtime_pivot_routes.py`

当前已覆盖：

- ReconWorker 真实命令执行与 probe adapter 解析
- `NmapAdapter` 文本输出解析
- `CustomProbeAdapter` JSON 输出解析
- phase-two result applier 对 runtime / KG / AG 的统一消费
- Recon / Access / Goal worker 的 `entity_upsert` / `relation_upsert` 产出
- worker 结果经 result applier 后形成正式 KG entity / relation 更新
- runtime credential / lease / pivot route manager 生命周期
- AccessWorker 的：
  - session 复用
  - session 缺失申请
  - credential 无效失败
  - pivot route 选择
  - privilege gap replan
- GoalWorker 的 satisfied / unsatisfied validator 路径

当前已确认的定向测试结果：

- `python -m pytest tests\\test_phase_two_workers.py tests\\test_agent_workers.py -q` 通过
- `python -m pytest tests\\test_phase_two_result_applier.py -q` 通过
- `python -m pytest tests\\test_agent_workers.py tests\\test_phase_two_workers.py tests\\test_phase_two_result_applier.py -q` 通过：`16 passed`
- `python -m pytest tests\\test_runtime_credentials.py tests\\test_runtime_leases.py tests\\test_runtime_pivot_routes.py -q` 通过：`12 passed`
- `python -m pytest tests\\test_runtime_sessions.py tests\\test_phase_two_result_applier.py -q` 通过：`7 passed`
- `python -m pytest tests\\test_phase_two_workers.py tests\\test_agent_workers.py -q` 通过：`18 passed`

说明：

- 这里记录的是第二阶段相关定向回归结果
- 不是新的全量测试统计值
- 后续如果再做第二阶段 manager / validator 接入，应继续以最新实际回归结果覆盖本节

### 13.16 第二阶段补充修改：KG / AG / TG 多主机规则增强

截至 2026-04-17，第二阶段又补了一轮与多主机场景直接相关的 KG / AG / TG 增量实现，目标是让 phase-two execution substrate 不再只覆盖 worker/runtime 结果流，还能把多主机横向路径正式沉淀到图模型和图构建规则里。

#### 13.16.1 KG 模型扩展

当前 KG 在已有 `Host / Service / Identity / Credential / Session / PrivilegeState / NetworkZone / Goal` 基础上，补充了多主机场景更常用的关系类型：

- `IDENTITY_AVAILABLE_ON`
- `REUSES_CREDENTIAL`
- `PRIVILEGE_SOURCE`
- `PIVOTS_TO`

对应含义：

- `IDENTITY_AVAILABLE_ON`
  - 表示某 identity 在某 host 上实际可用，而不只是“曾被观察到”
- `REUSES_CREDENTIAL`
  - 表示某 credential 可以在另一 host 上直接复用
- `PRIVILEGE_SOURCE`
  - 表示某 privilege state 的来源对象，例如 session / credential / identity
- `PIVOTS_TO`
  - 表示某 host 可以作为到另一 host 的 pivot / route source

这使得第二阶段的 KG 现在能更正式地表达：

- 网段 / 子网
  - 继续使用 `NetworkZone` 节点承载
- 凭据
  - 使用 `Credential` 节点
- 会话
  - 使用 `Session` 节点
- 跳板关系
  - 使用 `PIVOTS_TO` 边
- 主机到主机可达性
  - 继续使用 `CAN_REACH`
- 身份在主机上的可用性
  - 使用 `IDENTITY_AVAILABLE_ON`
- 权限状态来源
  - 使用 `PRIVILEGE_SOURCE`

#### 13.16.2 StateWriter 写入规则增强

`StateWriterAgent` 现已不再只把 observation / evidence 归一化成：

- `Observation` 节点
- `Evidence` 节点
- `OBSERVED_ON / SUPPORTED_BY / DERIVED_FROM`

同时还能从结构化工具结果中提取：

- `entities`
- `relations`

并把它们稳定转成正式 KG patch：

- `KGEntityPatch`
- `KGRelationPatch`

因此第二阶段现在已经支持以下写入链路：

`tool output -> observation/evidence payload -> StateWriter structured extraction -> KG entity/relation patches`

这意味着 recon / access / 其他工具只要把结果整理成标准化 `entities / relations` 结构，就能由 StateWriter 正式落入 KG，而不需要 worker 直接写图。

#### 13.16.3 AG 投影规则增强

`AttackGraphProjector` 已新增多主机横向移动相关状态：

- `IDENTITY_AVAILABLE_ON_HOST`
- `CREDENTIAL_REUSABLE_ON_HOST`
- `SESSION_ACTIVE_ON_HOST`
- `PRIVILEGE_SOURCE_KNOWN`
- `PIVOT_HOST_AVAILABLE`
- `LATERAL_SERVICE_EXPOSED`

并新增相应动作模板：

- `ESTABLISH_PIVOT_ROUTE`
- `REUSE_CREDENTIAL_ON_HOST`
- `EXPLOIT_LATERAL_SERVICE`

当前投影语义已经覆盖：

- 某 host 可作为 pivot
- 某 credential 可在另一 host 复用
- 某 service 暴露后会解锁 lateral service action
- 某 session 激活后会解锁新的 identity-context action

因此 AG 不再只是“单主机资产/服务验证图”，而开始具备多主机横移 planning substrate 的形态。

#### 13.16.4 TG 构建规则增强

`tg_builder.py` 当前也已补齐一批多主机调度语义：

- task candidate 会自动推导并归一化：
  - `host:*`
  - `credential:*`
  - `session:*`
  - `multi-host`
- 同 host 的任务默认会共享 host lock
- 同 credential / 同 session 的任务会共享 lock
- 相同 host 上会自动补 discovery -> validation -> lateral 的阶段依赖
- `estimated_cost / estimated_risk / estimated_noise` 会基于 host/session/credential/pivot 特征做细化调整

结果上，这一轮之后的 TG 已经更符合多主机场景：

- 同主机任务串行倾向更强
- 不同主机任务在无共享锁时更容易并行
- 同一凭据 / 同一会话具备冲突约束
- 横移前置发现链会自动形成依赖

### 13.17 第二阶段测试状态补充（KG / AG / TG）

截至 2026-04-17，这轮第二阶段新增或更新的测试文件包括：

- `tests/test_agents.py`
- `tests/test_ag_projector.py`
- `tests/test_tg_builder.py`

新增覆盖点：

- `StateWriterAgent` 从结构化 `entities / relations` 提取正式 KG patch
- KG 多主机关系类型：
  - `PIVOTS_TO`
  - `REUSES_CREDENTIAL`
  - 以及 subnet / credential / session 的结构化实体写入
- AG 多主机状态投影：
  - pivot host
  - credential reuse
  - session active on host
  - identity available on host
  - privilege source known
  - lateral service exposed
- AG 多主机动作投影：
  - `ESTABLISH_PIVOT_ROUTE`
  - `REUSE_CREDENTIAL_ON_HOST`
  - `EXPLOIT_LATERAL_SERVICE`
- TG 多主机调度规则：
  - 同 host 冲突
  - 不同 host 并行
  - credential / session lock
  - discovery -> lateral -> privilege 的阶段依赖

当前已确认的定向测试结果补充为：

- `python -m pytest tests\\test_agents.py tests\\test_ag_projector.py tests\\test_tg_builder.py -q` 通过：`22 passed`
- `python -m pytest tests\\test_phase_two_result_applier.py tests\\test_agent_workers.py -q` 通过：`9 passed`

说明：

- 这组结果反映的是第二阶段 KG / AG / TG 多主机增强后的定向回归
- 不是全仓全量测试
- 若后续再补统一 runtime orchestration、真实 access validator 或真实 goal validator，应继续覆盖更新本节

### 13.18 第二阶段补充修改：worker 结果协议收敛

截至 2026-04-17，第二阶段又补了一轮“结果协议收敛”工作，目标是解决当前仓库同时存在：

- `AgentOutput` 链路
- `AgentTaskResult` 链路

所带来的重复适配和多入口消费问题。

#### 13.18.1 Canonical result 明确为 `AgentTaskResult`

当前第二阶段已经正式把：

- `AgentTaskResult`

定义为 worker 结果的 canonical result。

对应语义调整为：

- `AgentOutput`
  - 继续保留为 agent 层通用 transport / step output
  - 适用于 pipeline 内部 step 间传递
- `AgentTaskResult`
  - 作为 worker 落地结果的正式协议
  - 供 orchestrator / phase-two result applier / runtime 落地链消费

因此后续第二阶段在结果消费层面的统一链路更准确应写作：

`worker AgentOutput -> AgentResultAdapter -> AgentTaskResult -> PhaseTwoResultApplier -> Runtime / StateWriter / GraphProjection`

而不再允许 pipeline、orchestrator、result applier 各自维护一份私有转换逻辑。

#### 13.18.2 新增集中式 `AgentResultAdapter`

在 `src/core/models/events.py` 中已新增：

- `AgentResultAdapter`

其职责是集中处理以下适配：

- `AgentTaskResult -> AgentTaskResult`
  - 直接透传
- `AgentExecutionResult -> AgentTaskResult`
  - 从 execution envelope 中提取 worker `AgentOutput` 并转换
- `AgentOutput -> AgentTaskResult`
  - 将 worker 证据、观察、outcome 摘要、错误状态统一归一化

当前 `AgentResultAdapter` 已负责：

- 读取 `agent_input.raw_payload["agent_task_result"]`
  - 若已有显式 canonical result，则直接使用
- 将 worker `evidence` 归一化为：
  - `EvidenceArtifact`
- 将 worker `observations` 归一化为：
  - `ObservationRecord`
- 根据 worker `errors` / `outcome.success`
  - 推导 `AgentResultStatus`
- 基于 `task_type`
  - 推导 `AgentRole`

这意味着第二阶段 worker 结果适配逻辑现在已经从：

- orchestrator 私有 helper
- result applier 私有入口判断

中收口到统一协议层。

#### 13.18.3 `AgentPipeline` 统一导出 canonical worker result

`AgentPipeline` 当前新增：

- `worker_task_results(...)`

它会把 execution cycle 中的：

- worker `PipelineStepResult`

统一转换成：

- `AgentTaskResult`

这样 pipeline 内部仍可继续使用：

- `AgentOutput`

作为通用 agent transport，但对外提供的 worker 落地结果已经固定为：

- `AgentTaskResult`

这让第二阶段后续的：

- orchestrator
- result applier
- runtime consumption

都不再需要关心 worker step 原始结构。

#### 13.18.4 `PhaseTwoResultApplier` 入口统一收敛

`PhaseTwoResultApplier.apply(...)` 当前已改成可以接受：

- `AgentTaskResult`
- `AgentExecutionResult`
- `AgentOutput`

但入口会立即调用：

- `AgentResultAdapter.to_task_result(...)`

把输入收敛成 canonical result，再继续处理：

- runtime effects
- StateWriter
- fact write conversion
- GraphProjection

因此 result applier 内部现在已经不再显式分支处理：

- `AgentOutput`
- `AgentTaskResult`

而是统一只处理 `AgentTaskResult`。

#### 13.18.5 Orchestrator 已复用统一适配层

上一轮 operation 主循环里，`AppOrchestrator` 还保留了一份本地的 worker step -> task result 转换逻辑。

本轮已改为直接调用：

- `pipeline.worker_task_results(execution)`

因此 orchestrator 也不再拥有自己的结果协议分叉。

这一步很关键，因为它意味着当前第二阶段在“主循环 + phase-two 落地”之间，已经基本形成了统一 worker 结果协议，而不是一边走 pipeline `AgentOutput`，另一边走 phase-two `AgentTaskResult` 的双轨实现。

### 13.19 第二阶段测试状态补充（结果协议收敛）

截至 2026-04-17，这轮第二阶段新增或更新的测试文件包括：

- `tests/test_phase_two_result_applier.py`
- `tests/test_agents.py`

新增覆盖点：

- `AgentResultAdapter` 作为集中式协议适配器的 canonical path
- `PhaseTwoResultApplier` 可直接接受 worker `AgentOutput`
- `PhaseTwoResultApplier` 入口会先收敛到 `AgentTaskResult`
- `AgentPipeline.worker_task_results(...)` 可从 worker step 稳定导出 canonical result
- orchestrator 复用 pipeline 统一适配层后的回归行为

当前已确认的定向测试结果补充为：

- `python -m pytest tests\\test_phase_two_result_applier.py -q` 通过：`4 passed`
- `python -m pytest tests\\test_phase_two_workers.py tests\\test_agent_workers.py -q` 通过：`18 passed`
- `python -m pytest tests\\test_agents.py -q` 通过：`15 passed`
- `python -m pytest tests\\test_app_orchestrator.py -q` 通过：`6 passed`

说明：

- 这组结果反映的是第二阶段“结果协议收敛”后的定向回归
- 不是全仓全量测试
- 若后续继续收敛 planner / critic / perception 与 operation 主循环之间的统一执行协议，应继续覆盖更新本节

### 13.20 第二阶段补充修改：KG / AG / TG 图版本闭环与正式 store apply

本轮修改把第二阶段中的图更新链路，从“主要产生 delta 供外部消费”推进到“具备最小持久化与版本闭环能力”。

1. KG store 侧补齐了图版本和正式 patch apply 入口

- `src/core/models/kg.py`
  - `GraphDelta` 新增：
    - `version`
    - `change_count`
    - `last_patch_batch_id`
    - `last_changed_at`
- `src/core/graph/kg_store.py`
  - `KnowledgeGraph` 新增：
    - `_version`
    - `_last_patch_batch_id`
    - `version` property
    - `last_patch_batch_id` property
    - `apply_patch_batch(request: dict[str, Any])`
  - `apply_patch_batch(...)` 现在可以直接消费 `StateWriter` 输出的序列化 patch batch：
    - 校验 `base_kg_version`
    - 依次应用 entity / relation upsert
    - 返回 `patch_batch_id / kg_version / applied ids`
  - `_record_change(...)` 会在每次结构变更时推进 KG version，并同步刷新 `GraphDelta` 的 version / change_count / last_changed_at
  - 新增 `_apply_entity_patch(...)`、`_apply_relation_patch(...)`、`_normalize_source_refs(...)`，用于把 agent 层 patch 正式落入 KG typed model

2. StateWriter 保持 KG owner，但现在有了正式 apply 入口

- `src/core/agents/state_writer.py`
  - `KGPatchApplyRequest` 新增：
    - `patch_batch_id`
    - `base_kg_version`
    - `resulting_kg_version`
  - `build_store_apply_request(...)` 现在会生成稳定的 patch batch id，并把当前 `kg_version` 写成 patch 基线版本
  - 新增 `apply_to_store(store: KnowledgeGraph, apply_request: KGPatchApplyRequest)`
    - 仍由 `StateWriter` 发起正式 store apply
    - KG store 只执行结构变更和版本推进
  - `KGDeltaEvent.metadata` 现在会附带：
    - `base_kg_version`
    - `patch_batch_id`
    - `store_apply_request`

这使得第二阶段下的 KG 写入边界保持不变：

- worker 不直接写 KG
- StateWriter 仍是正式 KG owner
- 但不再只是“吐出 patch 却没有正式落地入口”

3. AG 快照补齐来源 KG 版本锚点

- `src/core/models/ag.py`
  - `AttackGraph` 新增：
    - `_version`
    - `_source_kg_version`
    - `_projection_batch_id`
    - `_metadata`
  - 新增：
    - `version`
    - `source_kg_version`
    - `projection_batch_id`
    - `metadata`
    - `set_projection_metadata(...)`
  - `to_dict()/from_dict()` 现在会保留 AG metadata
- `src/core/graph/ag_projector.py`
  - `project(...)` 在完成投影后，会把当前 AG 显式锚定到：
    - `kg.version`
    - `kg.last_patch_batch_id`
    - `kg.delta.change_count`
  - `project_incremental(...)` 也会把合并后的 AG 继续锚到 fresh projection 的来源 KG version / projection batch

4. GraphProjection event 补齐版本元数据

- `src/core/agents/graph_projection.py`
  - `AGProjectionEvent` 新增：
    - `source_kg_version`
    - `ag_version`
    - `projection_batch_id`
  - `GraphProjectionAgent.execute(...)` 会：
    - 从 `KGEventBatch` 中提取来源 KG version
    - 计算本次 AG projection 的轻量版本号
  - `_build_projection_events(...)` 现在会把：
    - `source_kg_version`
    - `ag_version`
    - `projection_batch_id`
    - `projector_handoff`
    写入 emitted event metadata

说明：

- 当前阶段仍未引入正式 AG store
- 因此 `ag_version` 目前是轻量逻辑版本，用于追踪投影批次与来源 KG 版本
- 后续接入正式 AG store 时，可直接把这里替换成 store 返回版本号

5. TG 侧补齐最小版本与 frontier 标识

- `src/core/models/tg.py`
  - `TaskGraph` 新增：
    - `_version`
    - `_source_ag_version`
    - `_frontier_version`
    - `_metadata`
  - 新增：
    - `version`
    - `source_ag_version`
    - `frontier_version`
    - `set_metadata(...)`
  - `to_dict()/from_dict()` 现在会保留 TG metadata
- `src/core/graph/tg_builder.py`
  - `TaskGenerationResult` 新增：
    - `source_ag_version`
    - `tg_version`
    - `frontier_version`
  - `AttackGraphTaskBuilder.build_candidates(...)`
    - 会把 `graph.version` 写成 `source_ag_version`
    - 会生成稳定的 `frontier_version`
    - 会把这些元数据同步写回 `task_graph`
  - `TaskGraphBuilder.build_from_candidates(...)`
    - 在无 AG 上下文时也会生成最小 `frontier_version`
    - 会把 TG 当前版本号写入 `TaskGenerationResult`

这意味着现在已经形成了最小的来源图谱系：

- KG patch batch -> KG version
- KG version -> AG source_kg_version
- AG version -> TG source_ag_version
- TG 再通过 frontier_version 标识局部构建边界

6. 当前阶段的意义

这一轮不是在做完整数据库持久化系统，而是在第二阶段内完成“图写入与图投影的版本闭环骨架”：

- KG 已有正式 patch apply 入口
- AG 已有来源 KG 版本锚点
- TG 已有来源 AG 版本与 frontier 标识
- checkpoint / runtime store 后续可以直接基于这些 version/ref 做恢复与局部重算

### 13.21 第二阶段补充修改：图版本闭环相关测试补充

本轮针对 KG / AG / TG 版本闭环新增了定向测试覆盖：

1. `tests/test_agents.py`

- 新增 `test_state_writer_can_apply_patch_batch_to_kg_store_with_version_tracking`
  - 验证：
    - `StateWriter` 可生成正式 `KGPatchApplyRequest`
    - `apply_to_store(...)` 可真正把 entity / relation patch 落到 KG store
    - `patch_batch_id` 与 `resulting_kg_version` 可追踪
- 新增 `test_graph_projection_emits_versioned_projection_event`
  - 验证：
    - `GraphProjectionAgent` emitted event 会带 `source_kg_version`
    - `metadata.ag_version` 已生成

2. `tests/test_ag_projector.py`

- 新增 `test_projection_metadata_is_bound_to_source_kg_version`
  - 验证：
    - `AttackGraphProjector.project(...)` 返回的 AG 会绑定 `source_kg_version`
    - AG 序列化 metadata 中保留来源 KG version 与 change_count

3. `tests/test_tg_builder.py`

- 扩展 `test_ag_to_tg_builds_stable_tasks_and_dependencies`
  - 新增验证：
    - `TaskGenerationResult.source_ag_version == ag.version`
    - `TaskGenerationResult.tg_version == task_graph.metadata.version`
    - `TaskGenerationResult.frontier_version == task_graph.metadata.frontier_version`

当前已确认的定向测试结果补充为：

- `python -m pytest tests\\test_agents.py -q` 通过：`17 passed`
- `python -m pytest tests\\test_ag_projector.py -q` 通过：`5 passed`
- `python -m pytest tests\\test_tg_builder.py -q` 通过：`7 passed`

说明：

- 这组结果反映的是第二阶段“图存储与版本闭环”后的定向回归
- 不是全仓全量测试
- 后续若把 checkpoint、runtime store、operation recovery 正式绑定到 graph version，应继续扩展本节

### 13.22 第三阶段起步：PackyAPI / OpenAI-compatible LLM 底层接入

截至 2026-04-24，仓库已经不再是“只有 LLM 配置占位”。本轮先完成了 LLM 接入的第一步，只落地底层客户端封装，不改 planner / critic 的业务逻辑。

本轮新增文件：

- `src/core/agents/packy_llm.py`
- `tests/test_packy_llm.py`

#### 13.22.1 当前落地范围

本轮只完成以下能力：

- 封装 Packy / OpenAI-compatible 网关的最小客户端
- 统一从环境变量读取：
  - `AEGRA_LLM_API_KEY`
  - `AEGRA_LLM_BASE_URL`
  - `AEGRA_LLM_MODEL`
  - `AEGRA_LLM_TIMEOUT_SEC`
- 提供：
  - `PackyLLMConfig`
  - `PackyLLMClient`
  - `PackyLLMResponse`
  - `PackyLLMError`
- 支持：
  - `list_models()`
  - `complete_chat(...)`
- 对网关返回做两层文本提取：
  - 标准 `chat.completions` JSON 提取
  - 非标准 `data: {...}` SSE chunk 文本回退解析

本轮明确不做的事：

- 不把客户端直接注入 `PlannerAgent`
- 不把客户端直接注入 `CriticAgent`
- 不修改 orchestrator / pipeline 默认装配逻辑
- 不使用 `responses.create(...)`

#### 13.22.2 为什么第一步只接 `chat/completions`

这轮联调中，已经实测 PackyAPI 在当前账号/分组/模型下表现为：

- `GET /models` 可用
- `POST /chat/completions` 可用
- `POST /responses` 当前不稳定，返回：
  - `bad_response_status_code`

因此当前仓库的底层客户端约定为：

- 第一阶段接入只走 `chat/completions`
- 暂不基于 `responses` 做任何正式封装

这样可以把不稳定面限制在最底层，避免后续 advisor 层反复处理兼容性问题。

#### 13.22.3 PackyAPI 当前实测行为记录

对 `https://www.packyapi.com/v1` 的当前实测结论如下：

1. 模型列表：

- `models.list()` 可返回真实模型 ID，例如：
  - `gpt-5.2`
  - `gpt-5.3-codex`
  - `gpt-5.4`
  - `gpt-5.4-mini`
  - `gpt-5.5`

2. `responses` 行为：

- `client.responses.create(...)` 当前不可作为稳定入口
- 即使模型存在，也可能返回：
  - `400 bad_response_status_code`

3. `chat.completions` 行为：

- 请求可成功
- 但非 stream 请求下，网关仍可能返回 SSE 风格 chunk 文本
- 表现为 SDK 对象中的：
  - `choices[0].message.content == None`
- 真实内容却出现在：
  - `data: {"choices":[{"delta":{"content":"..."}}]}`
    这类 chunk 文本中

因此当前必须做的兼容处理是：

- 先尝试读取标准 JSON `message.content`
- 若为空，再回退解析整段 SSE 文本中的 `delta.content`

#### 13.22.4 当前 API 的使用方式

当前推荐使用的网关配置是：

```powershell
$env:OPENAI_API_KEY='你的 PackyAPI key'
$env:OPENAI_BASE_URL='https://www.packyapi.com/v1'
```

若要在项目内统一配置，优先使用：

```powershell
$env:AEGRA_LLM_API_KEY='你的 PackyAPI key'
$env:AEGRA_LLM_BASE_URL='https://www.packyapi.com/v1'
$env:AEGRA_LLM_MODEL='gpt-5.2'
```

说明：

- 项目内优先读 `AEGRA_*`
- 若未设置，再回退到 `OPENAI_*`
- 当前默认模型为：
  - `gpt-5.2`

#### 13.22.5 当前推荐的最小调用方式

项目内当前应通过：

- `PackyLLMConfig.from_env()`
- `PackyLLMClient.complete_chat(...)`

完成最小调用。

最小示例：

```python
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig

with PackyLLMClient(PackyLLMConfig.from_env()) as client:
    response = client.complete_chat(
        user_prompt="目标 127.0.0.1:8080 是 http，下一步建议做什么？",
        system_prompt="你是一个安全任务规划助手。",
    )
    print(response.text)
```

注意：

- 当前不要用 `responses.create(...)`
- 当前不要直接假设 `choices[0].message.content` 一定有值
- 当前应统一通过 `PackyLLMClient` 做调用，而不是在业务代码里散落网关兼容逻辑

#### 13.22.6 当前测试状态

本轮新增定向测试：

- `tests/test_packy_llm.py`

覆盖点：

- 标准 `chat.completions` JSON 文本提取
- SSE chunk 文本拼接提取
- `PackyLLMClient` 在网关返回 chunk 文本时的回退解析

当前已确认结果：

- `python -m pytest tests\\test_packy_llm.py -q` 通过：`3 passed`

#### 13.22.7 当前边界与下一步

当前状态应理解为：

- LLM 底层调用能力已经就位
- 但业务层还没有正式启用 LLM

也就是说：

- `PlannerAgent(llm_advisor=...)` 仍未接入实际 advisor
- `CriticAgent(llm_advisor=...)` 仍未接入实际 advisor
- orchestrator 默认 pipeline 仍不会自动调用 PackyAPI

下一步推荐顺序仍为：

1. 基于 `PackyLLMClient` 实现 `PackyPlannerAdvisor`
2. 在专用 pipeline / smoke 脚本中显式注入 `PlannerAgent(llm_advisor=...)`
3. 增加“无 key 回退 / 有 key 生效 / LLM 出错回退”的测试

### 13.23 Packy 集成第二步到第四步完成记录

截至 2026-04-24，PackyAPI 接入已经继续完成了第二步、第三步和第四步，当前状态不再只是“底层客户端可用”，而是已经具备：

- planner advisor 适配层
- 专用本地装配入口
- 基本回退与验证测试

#### 13.23.1 第二步：Planner advisor 适配层已落地

本轮新增文件：

- `src/core/agents/packy_planner_advisor.py`
- `tests/test_packy_planner_advisor.py`

新增能力：

- `PackyPlannerAdvisor`
- `PackyPlannerAdvisorConfig`

职责边界：

- 只把底层 LLM 文本结果转换成 `PlannerLLMAdvice`
- 只允许做候选排序建议和解释增强
- 不生成工具参数
- 不改 planner 主逻辑
- 不触发任务执行

当前实现细节：

- 基于 `PackyLLMClient.complete_chat(...)` 发起调用
- 会把：
  - `goal_ref`
  - `planning_context`
  - 候选 `PlanningCandidate`
  组织成 JSON prompt
- 只允许返回已有的 `candidate_id`
- `score_delta` 会按配置做钳制，避免模型建议过度覆盖 heuristic 打分
- 支持 fenced JSON / 普通 JSON / 嵌入文本中的第一段 JSON 容错解析
- 若网关异常或模型输出不可解析：
  - 安全回退为 `[]`
  - 不打崩 `PlannerAgent`

#### 13.23.2 第三步：专用装配入口已落地

本轮新增文件：

- `scratch_packy_planner_smoke.py`

该脚本是一个专用本地装配入口，明确不改默认 pipeline。

它当前做的事是：

- 构造一条最小 `KG -> AG -> PlannerInput` 链
- 通过 `PackyPlannerAdvisor.from_env()` 显式创建 advisor
- 用：
  - `PlannerAgent(llm_advisor=...)`
  进行注入
- 执行一次最小 planner 轮次
- 打印：
  - planner 是否成功
  - decision 数量
  - 首个 decision
  - 是否存在 `llm_advice`

这个脚本的作用是：

- 提供一个不影响默认 orchestrator / pipeline 的本地联调入口
- 在真正把 advisor 接到更高层装配前，先验证：
  - 环境变量
  - 模型调用
  - planner advice 写回

当前推荐运行方式：

```powershell
$env:AEGRA_LLM_API_KEY='你的 PackyAPI key'
$env:AEGRA_LLM_BASE_URL='https://www.packyapi.com/v1'
$env:AEGRA_LLM_MODEL='gpt-5.2'
python .\scratch_packy_planner_smoke.py
```

或兼容使用：

```powershell
$env:OPENAI_API_KEY='你的 PackyAPI key'
$env:OPENAI_BASE_URL='https://www.packyapi.com/v1'
python .\scratch_packy_planner_smoke.py
```

#### 13.23.3 第四步：回退与验证测试已补齐

本轮新增或扩展测试文件：

- `tests/test_packy_llm.py`
- `tests/test_packy_planner_advisor.py`
- `tests/test_packy_planner_smoke.py`

当前覆盖点包括：

1. 底层客户端：

- 标准 `chat.completions` JSON 文本提取
- SSE chunk 文本提取
- 网关返回 chunk 文本时的 fallback 解析
- `PackyLLMConfig.from_env()`：
  - 优先读取 `AEGRA_*`
  - 其次回退 `OPENAI_*`

2. Advisor 层：

- fenced JSON 解析
- 非法 `candidate_id` 过滤
- `score_delta` 钳制
- 网关异常时回退为空建议
- `PlannerAgent + PackyPlannerAdvisor` 的真实集成断言

#### 13.23.4 第五步：运行时默认装配已接入 planner advisor 开关

当前 `AppOrchestrator` 不再要求调用方总是手工传入 `pipeline`。

现在的默认行为是：

- `AppOrchestrator(settings=...)` 若未显式传入 `pipeline`
- 会自动调用：
  - `build_optional_agent_pipeline(...)`
- 统一装配标准 planner / task_builder / scheduler / critic

第一阶段接入完成后，运行时是否启用 planner LLM，不再依赖单独的脚本装配，而是由 orchestrator 默认装配逻辑决定。

#### 13.23.5 第六步：LLM 配置已收敛到 `AppSettings -> pipeline builder -> advisor`

第二阶段完成后，运行时主路径不再依赖 `PackyPlannerAdvisor.from_env()` 直接从环境变量散读配置。

当前推荐的主链路是：

1. `AppSettings.from_env()` 负责读取：

- `AEGRA_LLM_API_KEY`
- `AEGRA_LLM_BASE_URL`
- `AEGRA_LLM_MODEL`
- `AEGRA_LLM_TIMEOUT_SEC`
- `AEGRA_ENABLE_PLANNER_LLM_ADVISOR`

2. `AppSettings.to_packy_llm_config()` 负责把应用配置转换成：

- `PackyLLMConfig`

3. `AppOrchestrator._build_default_pipeline(...)` 负责：

- 从 `settings` 读取 `enable_planner_llm_advisor`
- 调用 `settings.to_packy_llm_config()`
- 把显式 `llm_client_config` 传入 `build_optional_agent_pipeline(...)`

4. `build_optional_agent_pipeline(...)` 现在支持：

- `llm_client_config: PackyLLMConfig | None`

当：

- `enable_packy_planner_advisor=True`
- 且传入了 `llm_client_config`

则 builder 会直接构造：

- `PackyPlannerAdvisor(client=PackyLLMClient(llm_client_config))`

只有在未显式提供 `llm_client_config` 的兼容路径下，才会继续回退到：

- `PackyPlannerAdvisor.from_env()`

也就是说，`from_env()` 现在保留为兼容/脚本场景入口，不再是运行时主路径。

#### 13.23.6 当前配置约束

当前运行时启用 planner advisor 的条件变为：

- `AppSettings.enable_planner_llm_advisor == True`
- 且 `AppSettings.llm_api_key` 非空

如果只打开开关但没有提供 `llm_api_key`，`AppOrchestrator` 会在默认 pipeline 装配阶段直接抛出明确错误，避免带着半配置状态启动。

当前 `AppSettings.to_packy_llm_config()` 的默认补全规则是：

- `base_url` 默认 `https://www.packyapi.com/v1`
- `model` 默认 `gpt-5.2`
- `timeout_sec` 默认 `30.0`

#### 13.23.7 第二阶段新增/更新测试点

本轮补充验证了：

1. `AppSettings.from_env()` 能完整读取 planner LLM 相关配置
2. `AppSettings.to_packy_llm_config()`：

- 有 key 时生成显式 `PackyLLMConfig`
- 无 key 时返回 `None`
- 缺省字段时回填 Packy 默认值

3. `build_optional_agent_pipeline(...)`：

- 在显式提供 `llm_client_config` 时不再调用 `PackyPlannerAdvisor.from_env()`
- 而是直接用显式 config 构造 advisor/client

4. `AppOrchestrator`：

- 未开启开关时默认不启用 planner advisor
- 开启开关且有 key 时，会把显式 `llm_client_config` 传给 builder
- 开启开关但没有 key 时，会抛出明确配置错误

第二阶段回归测试通过的组合是：

```powershell
python -m pytest tests\test_pipeline_builders.py tests\test_app_orchestrator.py tests\test_packy_llm.py -q
```

#### 13.23.8 第七步：`PackyCriticAdvisor` 已落地

第三阶段完成后，`CriticAgent` 不再只有接口占位，而是新增了真实可调用的：

- `src/core/agents/packy_critic_advisor.py`

其职责与 planner advisor 对称：

- 复用 `PackyLLMClient.complete_chat(...)`
- 只做 Critic finding 的失败归纳、摘要覆盖、解释增强
- 不生成新的 `finding_id`
- 不直接生成取消任务、替换任务、执行命令或图写入动作

它返回的结构是：

- `CriticLLMReview`

允许的字段只有：

- `finding_id`
- `summary_override`
- `rationale_suffix`
- `metadata`

如果网关异常、返回内容不可解析、或 `finding_id` 不在输入集合中：

- advisor 会安全回退为 `[]`
- 不影响 `CriticAgent.execute()` 主流程

#### 13.23.9 第八步：`CriticAgent` 已接入统一运行时装配链路

当前 `build_optional_agent_pipeline(...)` 已扩展支持：

- `critic_llm_advisor`
- `enable_packy_critic_advisor`

装配逻辑现在分两条并行可选链路：

1. planner：

- `enable_packy_planner_advisor`
- `planner_llm_advisor`

2. critic：

- `enable_packy_critic_advisor`
- `critic_llm_advisor`

当：

- `enable_packy_critic_advisor=True`
- 且传入了 `llm_client_config`

builder 会直接构造：

- `PackyCriticAdvisor(client=PackyLLMClient(llm_client_config))`

并注入：

- `CriticAgent(llm_advisor=resolved_critic_advisor)`

如果没有显式 `llm_client_config`，兼容路径仍会回退到：

- `PackyCriticAdvisor.from_env()`

#### 13.23.10 第九步：应用配置已增加 Critic LLM 开关

`AppSettings` 现在新增：

- `enable_critic_llm_advisor: bool = False`

并支持环境变量：

- `AEGRA_ENABLE_CRITIC_LLM_ADVISOR`

因此当前运行时默认装配条件变为：

- `enable_planner_llm_advisor`
- `enable_critic_llm_advisor`

两个开关都共享同一份：

- `llm_api_key`
- `llm_base_url`
- `llm_model`
- `llm_timeout_sec`

也就是说，第三阶段仍然保持“一套网关配置，多种 advisor 共享”的策略，没有拆分 planner/critic 的独立网关参数。

#### 13.23.11 当前默认装配约束

`AppOrchestrator._build_default_pipeline(...)` 现在会统一检查：

- 是否启用了 planner advisor
- 是否启用了 critic advisor
- 是否存在 `settings.to_packy_llm_config()`

如果任一 advisor 开关被打开，但缺少 `llm_api_key`，会在 orchestrator 默认装配阶段直接抛出清晰错误，避免进入半启用状态。

#### 13.23.12 第三阶段新增/更新测试点

本轮新增或扩展测试包括：

- `tests/test_packy_critic_advisor.py`
- `tests/test_pipeline_builders.py`
- `tests/test_app_orchestrator.py`
- `tests/test_agents.py`

当前已验证：

1. `PackyCriticAdvisor` 能解析 `chat.completions` 风格返回中的 JSON review
2. `PackyCriticAdvisor` 在网关错误时回退为空列表
3. `CriticAgent + PackyCriticAdvisor` 的真实集成能把 LLM 归纳写回 recommendation rationale
4. `build_optional_agent_pipeline(...)` 能显式注入 critic advisor
5. `build_optional_agent_pipeline(...)` 能通过 `enable_packy_critic_advisor=True` 装配 Packy critic advisor
6. `AppSettings.from_env()` 能读取 `AEGRA_ENABLE_CRITIC_LLM_ADVISOR`
7. `AppOrchestrator` 在启用 critic advisor 但未提供 `llm_api_key` 时会拒绝启动

第三阶段回归测试通过的组合是：

```powershell
python -m pytest tests\test_packy_critic_advisor.py tests\test_pipeline_builders.py tests\test_app_orchestrator.py tests\test_agents.py -q
```

#### 13.23.13 第十步：LLM advisor 可观测性已补齐

第四阶段没有继续扩展执行能力边界，而是补齐了“默认关闭但可观测”的治理信息。

当前 `AppOrchestrator` 会把统一的 LLM advisor 状态写入多个稳定位置：

1. `operation.execution.metadata["control_plane"]["llm_advisors"]`
2. `get_health_status()["llm_advisors"]`
3. `get_readiness_status()["llm_advisors"]`
4. `operation.execution.metadata["last_control_cycle"]["llm_advisors"]`

这些字段统一来自：

- `AppOrchestrator._llm_advisor_status(settings)`

当前暴露的结构是：

- `planner_enabled`
- `critic_enabled`
- `configured`
- `model`
- `base_url`

其中：

- `configured=False` 表示当前没有形成有效的 `PackyLLMConfig`
- 不会暴露 `api_key`

#### 13.23.14 当前治理语义

第四阶段之后，运行时上层可以不深入看 agent 实例，也能直接从 orchestrator 元数据判断：

1. planner advisor 是否被显式打开
2. critic advisor 是否被显式打开
3. 当前进程是否已具备可用的 LLM 网关配置
4. 当前默认会使用哪个 `model`
5. 当前默认会使用哪个 `base_url`

这让排查以下问题更直接：

- “为什么本轮没有走 planner/critic LLM”
- “为什么开关开了但仍然没生效”
- “当前运行时是不是拿到了正确的模型配置”

#### 13.23.15 第四阶段新增测试点

本轮主要扩展：

- `tests/test_app_orchestrator.py`

当前新增验证包括：

1. `create_operation(...)` 生成的 `control_plane.llm_advisors` 默认值
2. `get_health_status()` 会返回 `llm_advisors`
3. `get_readiness_status()` 会返回 `llm_advisors`
4. `run_operation_cycle(...)` 完成后，`last_control_cycle` 会记录 `llm_advisors`

第四阶段回归测试通过的组合是：

```powershell
python -m pytest tests\test_app_orchestrator.py -q
```

3. 专用装配入口：

- 缺失环境变量时，`scratch_packy_planner_smoke.py` 会给出明确错误信息
- 有 advisor 输出时，脚本会打印 `llm_advice`

#### 13.23.4 当前验证结果

当前已确认定向测试结果为：

- `python -m pytest tests\\test_packy_llm.py tests\\test_packy_planner_advisor.py tests\\test_packy_planner_smoke.py -q`
  - 通过：`11 passed`

补充说明：

- 这组结果反映的是 PackyAPI 接入第一步到第四步的定向回归
- 还不是默认 pipeline / orchestrator 的全局启用
- 也不代表所有运行路径已经开始自动调用 LLM

#### 13.23.5 当前状态总结

到这一轮为止，PackyAPI 接入已经完成：

1. 底层客户端封装
2. planner advisor 适配层
3. 专用装配入口
4. 回退与验证测试

但仍然保持以下边界：

- 默认 pipeline 不会自动注入 `PackyPlannerAdvisor`
- orchestrator 默认主循环不会自动调用 PackyAPI
- critic 侧仍未接入 Packy advisor

因此当前最准确的状态是：

- “Packy + Planner” 的最小闭环已经具备
- 但只在显式装配入口和定向测试里启用

下一步若继续推进，推荐顺序为：

1. 新增一个专用 pipeline 组装入口，把 `PackyPlannerAdvisor` 注入到 planner 阶段
2. 选择是否为 `CriticAgent` 增加对应的 Packy advisor
3. 再决定是否把 LLM 装配提升到 orchestrator 级别的可选配置

### 13.24 2026-04-28 LLM advisor 统一 decision 模型推进记录

本轮目标是把现有 LLM advisor 输出从“轻量 metadata 增强”推进为统一的结构化决策建议层，但不改变主流程控制权。

#### 13.24.1 新增统一模型

新增文件：

- `src/core/agents/llm_decision.py`
- `tests/test_llm_decision_models.py`

新增模型：

- `LLMDecisionStatus`
  - `accepted`
  - `rejected`
- `LLMDecisionSource`
  - `planner`
  - `critic`
- `LLMDecision`
- `LLMDecisionValidationResult`

当前 `LLMDecision` 是跨 planner / critic 的统一建议承载结构，用于表达：

- 来源 agent surface
- decision 类型
- 目标对象 ID 和类型
- score_delta
- rationale_suffix
- summary_override
- risk_notes
- replan_hint
- metadata

当前还新增了禁止字段检测：

- `contains_forbidden_llm_decision_key(...)`
- `FORBIDDEN_LLM_DECISION_KEYS`

用于拦截 LLM 输出中的直接工具命令、图写入 patch、任务取消/替换等越权意图。

#### 13.24.2 Planner 侧改造

修改文件：

- `src/core/agents/planner.py`
- `src/core/agents/packy_planner_advisor.py`
- `tests/test_packy_planner_advisor.py`

当前 Planner 侧保持原有 `PlannerLLMAdvice` 兼容接口，但新增：

- `decision: LLMDecision | None`
- `validation: LLMDecisionValidationResult | None`

`PackyPlannerAdvisor` 现在会把合法 advisor 输出标准化成：

- `LLMDecision(source=planner, decision_type=planner_candidate_advice, target_kind=planning_candidate, ...)`

仍然只允许：

- 引用已有 `candidate_id`
- 调整 `score_delta`
- 增加 `rationale_suffix`
- 增加 `risk_notes`
- 增加解释性 metadata

仍然不允许：

- 生成新的 candidate
- 生成工具命令
- 生成执行参数
- 直接修改 KG / AG / TG / Runtime

`PlannerAgent._apply_llm_advice(...)` 仍只做候选排序和解释增强，并把以下信息写入 candidate metadata：

- `llm_advice`
- `llm_decision`
- `llm_decision_validation`

#### 13.24.3 Critic 侧改造

修改文件：

- `src/core/agents/critic.py`
- `src/core/agents/packy_critic_advisor.py`
- `tests/test_packy_critic_advisor.py`

当前 Critic 侧保持原有 `CriticLLMReview` 兼容接口，但新增：

- `replan_hint: str | None`
- `decision: LLMDecision | None`
- `validation: LLMDecisionValidationResult | None`

`PackyCriticAdvisor` 现在会把合法 advisor 输出标准化成：

- `LLMDecision(source=critic, decision_type=critic_finding_review, target_kind=critic_finding, ...)`

仍然只允许：

- 引用已有 `finding_id`
- 覆盖 summary
- 补充 rationale
- 补充受限 `replan_hint`
- 增加解释性 metadata

仍然不允许：

- 生成新的 finding
- 直接取消任务
- 直接替换任务
- 生成工具命令
- 直接写 KG / AG / TG patch

`CriticAgent._apply_llm_review(...)` 仍只把 LLM review 合并进 finding 解释层，并把以下信息写入 finding metadata：

- `llm_review`
- `llm_decision`
- `llm_decision_validation`
- `llm_replan_hint`

真正的 cancel / replace / replan recommendation 仍由现有 Critic / TG / Runtime 规则生成。

#### 13.24.4 当前测试覆盖

本轮新增或扩展测试覆盖：

- 统一 `LLMDecision` Pydantic schema
- extra / 越权字段拒绝
- forbidden key 检测
- Planner 合法输出标准化
- Planner 未知 candidate 过滤
- Planner 越权工具字段过滤
- Planner 空输出 / 解析失败回退
- Critic 合法输出标准化
- Critic 未知 finding 过滤
- Critic 越权直接动作字段过滤
- Critic 空输出 / 解析失败回退
- fake client 路径，无真实网络调用

已确认定向回归通过：

```powershell
python -m pytest tests\test_llm_decision_models.py tests\test_packy_llm.py tests\test_packy_planner_advisor.py tests\test_packy_critic_advisor.py tests\test_agents.py tests\test_pipeline_builders.py -q
```

结果：

- `43 passed`

#### 13.24.5 当前边界

本轮没有让 LLM 接管主流程。

当前准确状态是：

- LLM advisor 输出已经进入统一结构化 decision 表达
- Planner / Critic 仍只消费受限建议
- 现有 KG / AG / TG / Runtime / Scheduler 主控边界没有改变
- worker 执行、工具命令生成、图状态写入仍不由 LLM 直接控制

后续如果继续推进，应在此基础上增加独立 validator / decision history，再考虑 SupervisorAgent 和受控自动循环。

### 13.25 2026-04-28 LLMDecisionValidator 接入记录

本轮目标是在 13.24 的统一 `LLMDecision` 模型基础上，新增统一校验层，确保 LLM advisor 输出必须先经过 schema、权限、引用对象、policy/runtime 约束校验，再被 Planner / Critic 消费。

#### 13.25.1 新增统一 validator

修改文件：

- `src/core/agents/llm_decision.py`
- `tests/test_llm_decision_models.py`

新增：

- `LLMDecisionValidator`

当前支持：

- `validate_planner_decision(...)`
- `validate_critic_decision(...)`
- `validate_no_forbidden_payload(...)`

校验结果统一返回：

- `LLMDecisionValidationResult`
  - `accepted`
  - `rejected`
  - `reason`
  - `sanitized_payload`

当前校验覆盖：

- planner / critic source 是否匹配
- decision_type 是否匹配
- target_kind 是否匹配
- candidate_id / finding_id 是否存在于本轮 agent 输入
- `score_delta` 是否越界
- 是否包含禁止字段
- 是否受 policy 禁用
- 是否受 runtime summary 禁用
- sanitized metadata 不保留 `api_key` / `secret` 类字段

禁止字段检测已从浅层 key 检查扩展为递归检查，能识别 metadata / nested payload 中的：

- `command`
- `shell_command`
- `tool_command`
- `tool_args`
- `patch`
- `state_delta`
- `kg_delta`
- `ag_delta`
- `tg_delta`
- `cancel_task`
- `replace_task`

#### 13.25.2 Planner 消费层接入

修改文件：

- `src/core/agents/planner.py`
- `src/core/agents/packy_planner_advisor.py`
- `tests/test_packy_planner_advisor.py`
- `tests/test_agents.py`

当前 Planner 侧有两层校验：

1. `PackyPlannerAdvisor` 解析模型输出时：

- 先调用 `validate_no_forbidden_payload(...)` 拒绝直接命令 / 图 patch / 任务动作等越权字段
- 再构造 `LLMDecision`
- 再调用 `validate_planner_decision(...)`
- 只有 accepted 的 advice 才返回给 `PlannerAgent`

2. `PlannerAgent._apply_llm_advice(...)` 消费 advice 前：

- 对所有 advisor 返回项重新构造或读取 `LLMDecision`
- 调用 `validate_planner_decision(...)`
- rejected advice 不进入候选排序、不改变 rationale、不写入 candidate metadata
- logs 记录：
  - `planner llm decision validation accepted=X rejected=Y`

因此即使是非 Packy 的自定义 advisor，也不能绕过 Planner 消费层校验。

#### 13.25.3 Critic 消费层接入

修改文件：

- `src/core/agents/critic.py`
- `src/core/agents/packy_critic_advisor.py`
- `tests/test_packy_critic_advisor.py`
- `tests/test_agents.py`

当前 Critic 侧同样有两层校验：

1. `PackyCriticAdvisor` 解析模型输出时：

- 先调用 `validate_no_forbidden_payload(...)`
- 再构造 `LLMDecision`
- 再调用 `validate_critic_decision(...)`
- 只有 accepted 的 review 才返回给 `CriticAgent`

2. `CriticAgent._apply_llm_review(...)` 消费 review 前：

- 对所有 advisor 返回项重新构造或读取 `LLMDecision`
- 调用 `validate_critic_decision(...)`
- rejected review 不改变 finding summary / rationale / metadata
- logs 记录：
  - `critic llm decision validation accepted=X rejected=Y`

因此即使是非 Packy 的自定义 critic advisor，也不能直接通过 review 携带工具命令、图 patch、cancel / replace 动作。

#### 13.25.4 当前 policy / runtime 约束

当前 validator 支持通过 policy/runtime 显式禁用 LLM decision：

- `policy_context["disable_llm_decisions"]`
- `policy_context["llm_decisions_disabled"]`
- `runtime_summary["disable_llm_decisions"]`
- `runtime_summary["llm_decisions_disabled"]`

当这些开关存在并为真时，validator 返回 rejected，不影响 planner / critic 确定性主逻辑。

#### 13.25.5 当前测试覆盖

本轮新增或扩展测试覆盖：

- validator accepted
- validator rejected
- sanitized metadata
- nested forbidden key 检测
- policy 禁用
- runtime 禁用
- planner unknown candidate 拒绝
- planner score_delta 越界拒绝
- planner 消费层 rejected 不影响候选
- critic forbidden metadata 拒绝
- critic unknown finding 拒绝
- critic 消费层 rejected 不影响 finding / recommendation
- advisor gateway 异常回退仍保持为空建议

定向回归通过：

```powershell
python -m pytest tests\test_llm_decision_models.py tests\test_packy_llm.py tests\test_packy_planner_advisor.py tests\test_packy_critic_advisor.py tests\test_agents.py tests\test_pipeline_builders.py -q
```

结果：

- `48 passed`

#### 13.25.6 当前边界

本轮仍未让 LLM 接管主流程。

当前准确状态是：

- LLM 输出必须经过统一 validator
- 被拒绝的输出不会影响 planner / critic 主逻辑
- agent logs 已记录 accepted / rejected 统计
- API key 不会写入 sanitized metadata
- KG / AG / TG / Runtime / Scheduler 的主控边界保持不变

下一步如果继续推进，应补统一 `llm_decision_history`，把 accepted / rejected 的结构化摘要持久写入 operation metadata，而不仅仅是 agent logs。

### 13.26 2026-04-28 Planner LLM 受限策略参与者推进记录

本轮目标是在不改变 Planner 主流程控制权的前提下，让 LLM 从“逐候选解释增强”升级为 Planner 内的受限策略参与者。

#### 13.26.1 新增 PlannerLLMDecision schema

修改文件：

- `src/core/agents/planner.py`
- `src/core/agents/llm_decision.py`
- `tests/test_llm_decision_models.py`

新增 planner 侧 schema：

- `PlannerLLMRankAdjustment`
- `PlannerLLMDecision`

`PlannerLLMDecision` 当前允许字段：

- `selected_candidate_ids`
- `rank_adjustments`
- `risk_notes`
- `defer_reason`
- `requires_human_review`
- `metadata`
- `decision`
- `validation`

其中 `rank_adjustments` 只能引用已有 `candidate_id`，并且只能包含：

- `score_delta`
- `rationale_suffix`
- `risk_notes`
- `metadata`

仍然不允许：

- 生成新 candidate
- 修改 task payload
- 生成工具参数
- 生成 shell / tool command
- 写 KG / AG / TG / Runtime patch

#### 13.26.2 Validator 扩展

`LLMDecisionValidator` 新增：

- `validate_planner_strategy_decision(...)`

当前校验内容包括：

- strategy envelope 必须是：
  - `source=planner`
  - `decision_type=planner_strategy_decision`
  - `target_kind=planner_goal`
- `target_id` 必须属于本轮 goal refs
- `selected_candidate_ids` 必须全部来自本轮候选
- `rank_adjustments[].candidate_id` 必须来自本轮候选
- `rank_adjustments[].score_delta` 不得超过阈值
- policy / runtime 禁用开关仍然生效
- forbidden payload 递归检查仍然生效
- sanitized metadata 不保留 `api_key` / `secret` 类字段

#### 13.26.3 PackyPlannerAdvisor 迁移

修改文件：

- `src/core/agents/packy_planner_advisor.py`
- `tests/test_packy_planner_advisor.py`

`PackyPlannerAdvisor.advise(...)` 当前优先返回：

- `PlannerLLMDecision`

新的模型输出 schema 从旧的：

- `{"advice": [...]}`

升级为：

- `selected_candidate_ids`
- `rank_adjustments`
- `risk_notes`
- `defer_reason`
- `requires_human_review`
- `metadata`

为了兼容旧测试和旧模型输出，当前仍支持 legacy `advice` 数组，并将其转换成一个 `PlannerLLMDecision`：

- legacy `candidate_id` -> `rank_adjustments[].candidate_id`
- legacy `score_delta` -> `rank_adjustments[].score_delta`
- legacy `rationale_suffix` -> `rank_adjustments[].rationale_suffix`
- legacy `risk_notes` -> `rank_adjustments[].risk_notes`

也就是说，Packy advisor 的主接口已经迁移到 planner-level strategy decision，但旧 per-candidate advice 仍有兼容路径。

#### 13.26.4 PlannerAgent 消费策略 decision

修改文件：

- `src/core/agents/planner.py`
- `tests/test_packy_planner_advisor.py`
- `tests/test_agents.py`
- `tests/test_pipeline_builders.py`

`PlannerAgent._apply_llm_advice(...)` 当前支持两类返回：

1. 新路径：

- `PlannerLLMDecision`

2. 兼容路径：

- `list[PlannerLLMAdvice]`

当收到 `PlannerLLMDecision` 时，PlannerAgent 会：

- 构造或读取 `LLMDecision` envelope
- 调用 `validate_planner_strategy_decision(...)`
- 如果 rejected：
  - 完全回退到 heuristic candidates
  - 不改变排序
  - 不改变 rationale
  - 不写入 candidate metadata
  - logs 记录 rejected reason
- 如果 accepted：
  - 根据 `rank_adjustments` 做受限 `score_delta`
  - 根据 `selected_candidate_ids` 做小幅确定性排序 nudging
  - 根据 `rationale_suffix` 增强 rationale
  - 不修改 task payload
  - 不生成任何任务、工具命令或执行动作

Planner output metadata 当前会写入：

- `llm_decision_summary`
  - `adopted`
  - `reason`
  - `selected_candidate_ids`
  - `requires_human_review`
  - `defer_reason`
- `llm_planner_decision`
- `llm_decision`
- `llm_decision_validation`

#### 13.26.5 当前测试覆盖

本轮新增或迁移测试覆盖：

- `PlannerLLMDecision` legacy advice 迁移
- Packy advisor 返回 planner-level strategy decision
- forbidden tool / graph 字段拒绝
- LLM 改变候选排序
- LLM 要求人工复核
- 非法 candidate 被拒绝
- rejected 后完全回退 heuristic candidates
- Packy gateway 异常回退为空建议
- PlannerAgent 不产生 state_deltas / 不 dispatch
- `LLMDecisionValidator.validate_planner_strategy_decision(...)` accepted / rejected

定向回归通过：

```powershell
python -m pytest tests\test_llm_decision_models.py tests\test_packy_llm.py tests\test_packy_planner_advisor.py tests\test_packy_critic_advisor.py tests\test_agents.py tests\test_pipeline_builders.py -q
```

结果：

- `51 passed`

#### 13.26.6 当前边界

本轮仍未让 LLM 接管主流程。

当前准确状态是：

- LLM 已能在 PlannerAgent 内作为受限策略参与者影响候选排序和解释
- LLM 只能在已有 candidates 集合内选择、排序、标注风险和请求人工复核
- LLM 不能新增 candidate
- LLM 不能修改 task payload
- LLM 不能生成工具命令
- LLM 不能直接执行
- Planner 的候选生成、task export、TG 构建、Scheduler、Worker 执行边界都没有改变

下一步如果继续推进，应在 Critic 侧增加受限 replan proposal schema，但仍保持 Critic 只产出 recommendation / ReplanRequest metadata，不直接改 TG / KG / AG。

### 13.27 2026-04-28 Critic LLM 受限 replan proposal 推进记录

本轮目标是在不改变 Critic / TG / Runtime 主控边界的前提下，让 LLM 辅助 Critic 产生受限 replan proposal。

#### 13.27.1 当前相关实现边界复核

本轮查看并确认了以下模块：

- `src/core/agents/critic.py`
- `src/core/planner/critic.py`
- `src/core/models/runtime.py`
- `src/core/agents/agent_models.py`
- `src/core/agents/packy_critic_advisor.py`

当前主边界保持不变：

- `TaskGraphCritic` 仍负责确定性 TG finding / frontier 规则
- `CriticAgent` 仍只产出 recommendation / replan request / advisory state delta
- `RuntimeState.replan_requests` 和 `ReplanRequest.metadata` 可承载后续 runtime 层 metadata
- `ReplanRequestRecord.payload` 是 agent 输出层承载 replan metadata 的当前位置
- LLM 不直接调用 `cancel_task(...)`
- LLM 不直接调用 `mark_task_superseded(...)`
- LLM 不直接调用 `replace_subgraph(...)`
- LLM 不直接写 KG / AG / TG patch

#### 13.27.2 新增 CriticLLMReplanProposal schema

修改文件：

- `src/core/agents/critic.py`

新增：

- `CriticLLMReplanProposal`

允许字段：

- `finding_id`
- `failure_summary`
- `replan_hint`
- `affected_task_ids`
- `confidence`
- `requires_human_review`
- `metadata`
- `decision`
- `validation`

该 schema 只能绑定到已有 `CriticFinding`，不能独立生成新的 finding、task、patch 或 action。

`CriticLLMReview` 现在新增：

- `replan_proposal: CriticLLMReplanProposal | None`

也就是说，LLM 的 replan 建议仍作为 finding review 的附属信息进入 Critic，而不是成为独立执行动作。

#### 13.27.3 Validator 扩展

修改文件：

- `src/core/agents/llm_decision.py`
- `tests/test_llm_decision_models.py`

`LLMDecisionValidator` 新增：

- `validate_critic_replan_proposal(...)`

当前校验内容包括：

- decision envelope 必须是：
  - `source=critic`
  - `decision_type=critic_replan_proposal`
  - `target_kind=critic_finding`
- `finding_id` 必须存在于本轮 findings
- `affected_task_ids` 必须来自本轮 finding 的 TG subject refs
- `confidence` 必须在 `[0, 1]`
- policy / runtime 禁用开关仍然生效
- forbidden payload 递归检查仍然生效
- sanitized metadata 不保留 `api_key` / `secret` 类字段

仍然禁止：

- shell 命令
- 工具参数
- 图写入 patch
- `cancel_task`
- `replace_task`
- KG / AG / TG delta

#### 13.27.4 PackyCriticAdvisor 扩展

修改文件：

- `src/core/agents/packy_critic_advisor.py`
- `tests/test_packy_critic_advisor.py`

Packy critic prompt 的 response schema 现在允许 review 中包含：

- `failure_summary`
- `replan_hint`
- `affected_task_ids`
- `confidence`
- `requires_human_review`

`PackyCriticAdvisor` 解析后会构造：

- `CriticLLMReview`
- 可选 `CriticLLMReplanProposal`
- 对应 `LLMDecision(decision_type=critic_replan_proposal)`

Packy 层仍然先执行：

- `validate_no_forbidden_payload(...)`

因此模型输出中只要包含直接动作或图 patch 字段，会被过滤，不进入 CriticAgent。

#### 13.27.5 CriticAgent 消费与 metadata 写入

修改文件：

- `src/core/agents/critic.py`
- `tests/test_agents.py`
- `tests/test_packy_critic_advisor.py`

`CriticAgent._apply_llm_review(...)` 现在会：

1. 校验常规 `CriticLLMReview`
2. 若存在 `replan_proposal`，再调用 `validate_critic_replan_proposal(...)`
3. 若 proposal rejected：
   - 不改变 finding
   - 不影响 recommendation
   - 不影响 replan request
4. 若 proposal accepted：
   - 写入 finding metadata：
     - `llm_replan_proposal`
     - `runtime_metadata.llm_replan_proposal`
   - 后续 recommendation rationale 会追加受限 replan hint
   - recommendation metadata 会包含 `llm_replan_proposal`
   - ReplanRequestRecord payload 会包含：
     - `llm_replan_proposal`
     - `runtime_metadata.llm_replan_proposal`

当前 Runtime metadata 记录的是 agent output payload 中的结构化 runtime metadata，供后续 result applier / runtime store 消费；CriticAgent 本身仍不直接修改 `RuntimeState.execution.metadata`。

#### 13.27.6 当前 replan 主控边界

真正是否产生 replan request 仍由现有规则决定：

- finding type 必须属于：
  - `permanently_blocked_tasks`
  - `repeated_failures`
  - `invalidated_by_new_evidence`
  - `low_value_paths`
- 必须存在 TG task subject refs
- frontier 仍由 `TaskGraph.collect_replan_frontier(...)` / `TaskGraphCritic.collect_replan_frontier(...)` 决定

LLM proposal 只能影响：

- recommendation rationale
- replan request payload metadata
- confidence hint
- human review hint

LLM proposal 不能：

- 直接 cancel task
- 直接 replace task
- 直接创建 new task
- 直接修改 TG
- 直接写 KG / AG
- 直接执行工具

#### 13.27.7 当前测试覆盖

本轮新增或扩展测试覆盖：

- 合法 critic replan proposal accepted
- unknown task_id rejected
- 合法 Packy critic 输出生成 `CriticLLMReplanProposal`
- 越权字段如 `cancel_task` 被过滤
- gateway 异常仍回退为空 review
- proposal 写入 recommendation metadata
- proposal 写入 ReplanRequestRecord payload runtime metadata
- proposal replan_hint 被合并进 recommendation rationale

定向回归通过：

```powershell
python -m pytest tests\test_llm_decision_models.py tests\test_packy_llm.py tests\test_packy_planner_advisor.py tests\test_packy_critic_advisor.py tests\test_agents.py tests\test_pipeline_builders.py -q
```

结果：

- `55 passed`

#### 13.27.8 当前边界

本轮仍未让 LLM 接管 Critic 或 replan 主流程。

当前准确状态是：

- LLM 可以为 Critic finding 附加受限 replan proposal
- proposal 必须绑定已有 finding
- proposal affected task 必须来自本轮 finding 的 TG refs
- proposal 只进入 rationale / metadata / ReplanRequest payload
- 现有 Critic / TG / Runtime 规则仍决定是否 cancel / replace / replan

下一步如果继续推进，应实现统一 `llm_decision_history`，把 Planner / Critic 的 accepted / rejected 结构化摘要从 logs / payload metadata 收敛进 operation metadata 查询面。

### 13.28 2026-04-28 SupervisorAgent 骨架与可选装配记录

本轮查看并确认了 `AgentPipeline`、`AgentKind`、`registry` 和 `AppOrchestrator.run_operation_cycle(...)` 的编排边界：

- `run_operation_cycle(...)` 当前主流程仍是：
  - planning
  - execution
  - apply
  - feedback / critic
  - persist cycle summary
- `AgentRegistry` 只按 kind/name 注册和分发 agent，不隐含执行顺序。
- `AgentPipeline` 原有主 cycle 不包含 supervisor。
- `last_control_cycle` 由 orchestrator 的 `_persist_cycle_summary(...)` 写入 runtime metadata。

#### 13.28.1 新增文件与模型

新增：

- `src/core/agents/supervisor.py`
- `src/core/agents/packy_supervisor_advisor.py`

新增 `AgentKind.SUPERVISOR`。

Supervisor 侧新增结构：

- `SupervisorStrategy`
  - `continue_planning`
  - `continue_execution`
  - `request_replan`
  - `pause_for_review`
  - `stop_when_quiescent`
- `SupervisorContext`
  - `runtime_summary`
  - `last_control_cycle`
  - `planner_summary`
  - `critic_summary`
  - `budget_summary`
- `SupervisorDecision`
  - `strategy`
  - `rationale`
  - `confidence`
  - `requires_human_review`
  - `metadata`
- `SupervisorLLMAdvisor`

同时扩展 `LLMDecisionSource.SUPERVISOR`，为后续统一 LLM decision history 留出 source 枚举。

#### 13.28.2 SupervisorAgent 当前职责

`SupervisorAgent` 当前只是高层策略建议 agent：

- 输出 `DecisionRecord(decision_type="supervisor_strategy")`
- 只写入 `AgentOutput.decisions` 和 `logs`
- payload 中包含：
  - `supervisor_decision`
  - `llm_decision_validation`
  - `llm_adopted`
  - `control_only`

当前明确禁止：

- 直接写 KG / AG / TG / Runtime
- 直接执行 worker
- 直接发工具命令
- 直接产生 `state_deltas`
- 直接产生 `replan_requests`
- 直接 cancel / replace task

LLM advisor 输出会经过禁止字段检查，包含 `command`、`tool_command`、`patch`、`kg_delta`、`ag_delta`、`tg_delta`、`cancel_task`、`replace_task` 等字段时会 rejected，并回退到 heuristic supervisor decision。

#### 13.28.3 可选 pipeline 装配

`AgentPipelineAssemblyOptions` 新增：

- `enable_packy_supervisor_advisor`
- `include_supervisor`

`build_optional_agent_pipeline(...)` 新增：

- `supervisor_llm_advisor`

默认行为：

- 不注册 `SupervisorAgent`
- 不启用 `PackySupervisorAdvisor`
- 不改变 planner / scheduler / worker / critic 流程

只有以下情况会注册 supervisor：

- `include_supervisor=True`
- `enable_packy_supervisor_advisor=True`
- 显式传入 `supervisor_llm_advisor`

`AgentPipeline` 新增独立的 `run_supervisor_cycle(...)`，用于后续外部显式调用 supervisor advisory cycle；当前 orchestrator 主循环没有调用它。

#### 13.28.4 PackySupervisorAdvisor

新增 `PackySupervisorAdvisor`，默认关闭。

它只解析 JSON 形式的 `SupervisorDecision`，prompt 明确约束只能返回 supervisor strategy，不允许任务、命令、工具参数、图 patch 或 cancel / replace 动作。

测试中使用 fake client，不启用真实网络调用。

#### 13.28.5 AppSettings / Orchestrator 状态

`AppSettings` 新增：

- `enable_supervisor_llm_advisor`
- 环境变量：`AEGRA_ENABLE_SUPERVISOR_LLM_ADVISOR`

`AppOrchestrator._build_default_pipeline(...)`：

- 当 supervisor advisor 开关打开时，会把 `enable_packy_supervisor_advisor=True` 传给 pipeline builder
- 若打开任一 LLM advisor 但没有 `llm_api_key`，仍会 fail fast

`_llm_advisor_status(...)` 新增：

- `supervisor_enabled`

`run_operation_cycle(...)` 当前没有插入 supervisor phase，因此 orchestrator 行为不变。

#### 13.28.6 测试覆盖

新增 `tests/test_supervisor_agent.py`，覆盖：

- 默认 pipeline 不启用 supervisor
- 显式启用后输出结构化策略建议
- 非法 advisor 建议包含禁止字段时 rejected 并回退
- `PackySupervisorAdvisor` 使用 fake client 解析合法 JSON
- 即使 pipeline 注册了 `SupervisorAgent`，orchestrator 主循环也不会执行 supervisor step

同步更新：

- `tests/test_app_orchestrator.py`
- `tests/test_vulhub_orchestrator_smoke.py`

定向回归：

```powershell
python -m pytest tests\test_supervisor_agent.py tests\test_pipeline_builders.py tests\test_app_orchestrator.py -q
```

结果：

- `33 passed`

全量回归：

```powershell
python -m pytest -q
```

结果：

- `219 passed, 5 skipped`

#### 13.28.7 当前准确状态

Supervisor 已进入“可选注册 + 独立 advisory cycle + 结构化策略建议”阶段。

它还没有接管主流程：

- 不改变 `run_operation_cycle(...)`
- 不改变 task selection
- 不改变 scheduler / worker execution
- 不改变 critic replan 决策
- 不写任何图状态

后续若继续推进多智能体协作层，建议下一段先做：

- 统一 `llm_decision_history`
- 将 Planner / Critic / Supervisor 的 adopted / rejected decision summary 汇总进 operation metadata
- 再让 orchestrator 在 cycle summary 后可选调用 `run_supervisor_cycle(...)` 记录策略建议，但仍不改变主流程分支

### 13.29 2026-04-28 LLM decision history 可观测记录

本轮查看并确认：

- `RuntimeState.execution.metadata` 是 operation 级控制面 metadata 的落点。
- `operation_log` / `audit_log` 由 `src/core/runtime/observability.py` 写入，并已有脱敏与裁剪逻辑。
- `AppOrchestrator.run_operation_cycle(...)` 的 cycle summary 当前仍由 `_persist_cycle_summary(...)` 负责。
- Planner / Critic / Supervisor 的 LLM validation 已经存在于 agent output 的轻量 metadata 中，适合在 orchestrator 层汇总，而不是保存原始 prompt 或完整模型响应。

#### 13.29.1 新增统一 history 结构

新增文件：

- `src/core/runtime/llm_history.py`

新增：

- `LLM_DECISION_HISTORY_KEY = "llm_decision_history"`
- `LLMDecisionHistoryRecord`
- `ensure_llm_decision_history(...)`
- `append_llm_decision_history(...)`
- `recent_llm_decision_history(...)`

每条 history 记录包含：

- `cycle_index`
- `agent_kind`
- `advisor_type`
- `enabled`
- `configured`
- `decision_type`
- `accepted`
- `rejected_reason`
- `model`
- `created_at`

当前不保存：

- API key
- 原始 prompt
- 原始 LLM response
- 长文本上下文

#### 13.29.2 Orchestrator 接入

`create_operation(...)` 现在初始化：

- `execution.metadata["llm_decision_history"] = []`

`run_operation_cycle(...)` 在以下阶段后自动扫描 agent output 并追加 history：

- planning completed
- feedback completed

扫描来源包括：

- `AgentOutput.decisions`
- `AgentOutput.replan_requests`
- agent logs 中的 rejected reason

`_persist_cycle_summary(...)` 新增：

- `llm_decision_history_count`

health/readiness 仍只暴露简洁状态：

- advisor enabled/configured/model/base_url
- 不展开 history 明细

#### 13.29.3 查询入口

`AppOrchestrator` 新增：

- `get_llm_decision_history(operation_id, limit=20)`
- `record_llm_decision_cycle(operation_id, cycle_index, cycle)`

第二个方法用于显式记录独立 pipeline cycle，例如当前还没有进入主循环的 `run_supervisor_cycle(...)` 输出。

API 新增：

- `GET /operations/{operation_id}/llm-decisions?limit=N`

返回最近 N 条 operation-level LLM decision history。

#### 13.29.4 Critic metadata 补充

为保证 Critic accepted decision 也能被统一扫描，本轮补充了 recommendation metadata：

- `llm_decision`
- `llm_decision_validation`
- `llm_review`

同时 Critic rejected validation 会写入 agent logs：

- `critic llm decision rejected: <reason>`

这样 rejected 决策即使没有进入 recommendation，也能形成可审计 history。

#### 13.29.5 测试覆盖

新增：

- `tests/test_llm_decision_history.py`

覆盖：

- 默认空历史
- planner decision 记录
- critic decision 记录
- supervisor decision 记录
- rejected reason 记录
- 最近 N 条查询
- 不泄露 API key

定向回归：

```powershell
python -m pytest tests\test_llm_decision_history.py tests\test_app_orchestrator.py tests\test_supervisor_agent.py -q
```

结果：

- `33 passed`

全量回归：

```powershell
python -m pytest -q
```

结果：

- `225 passed, 5 skipped`

#### 13.29.6 当前准确状态

当前已经形成 Planner / Critic / Supervisor 的统一可观测 LLM decision history。

边界仍然保持：

- history 只记录审计摘要
- 不改变 agent 主逻辑
- 不让 LLM 接管 orchestrator 分支
- 不保存 prompt 全文或 API key
- Supervisor 仍需显式 cycle 才会进入 history

下一步如果继续推进，应考虑把 `llm_decision_history` 和 `operation_log` / `audit_log` 的导出视图合并成统一 audit report section，并为 UI/API 增加 agent_kind / accepted 过滤参数。
