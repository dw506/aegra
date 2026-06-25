# Aegra 重构设计 —— Agentic Planner + 三层图 + 单执行器 PEV 循环

> 状态：正式提案 (Proposed) · 版本：v3
> 范围：主控制循环 / Agent 模型 / 图模型（KG·AG·Log 三层）/ 写图链路 / 成功契约
> 取代：`aegra_plan_execute_verify_refactor.md`、`aegra_multihost_graph_driven_refactor.md`（均废弃删除）；旧的 `aegra_two_graph_architecture.md` / `aegra_runtime_flow.md` / `graph_driven_multi_agent_architecture.md` / `aegra_llm_stage_architecture.md` 已删除（描述的是被取代的多 stage-agent 架构）。本文档为当前架构的权威来源，待重写为正式版。

设计基调：**厚 planner + 薄执行器的单层 Plan→Execute→Verify 循环**。把决策权全部收回 planner；
执行器退化为"单目标一轮"的能力 agent；图拆成 **KG（状态真相）/ AG（结果叙事）/ Log（过程明细）** 三层；
写图与控制流彻底解耦；成功由**声明式契约 + 确定性评估**判定，planner 只读判决不判赢。
全程不绑定任何具体环境（full_chain_lab 仅为验证样例）。保留两条授权渗透不变量：**授权 scope 边界**、**密钥不落地**。

---

## 0. 重构目标（一句话）

把"每 cycle Planner 单选一个 StageAgent → StageAgent 内部再跑一整个 ReAct 自判 finish/replan → 回头重判"的**双层 LLM 反复横跳**，收敛为：

```text
Plan（planner 用图工具读图+判断）
  → Execute（单执行器跑完一个目标，过程落 log，回传结果）
  → Verify/Write（确定性抽事实写 KG + planner typed 工具写 AG 结果节点 + 契约评估刷新成功门）
  → 继续 / 重规划 / 终止
```

---

## 1. 背景与三个结构性症结

被取代的旧主链路（多 stage-agent 模型）：

```
KG/AG/Runtime → PlannerAgent(单选 stage) → ResultApplier → StageDispatcher
→ 1 个 StageAgent(内部 ReAct 自判) → MCP → ExecutionResult
→ AttackLogExtractor → ResultApplier → KG/AG
```

| 症结 | 根因 | 代码证据 |
|---|---|---|
| **AG 记录太细 → 节点膨胀** | 每次工具调用都生成一个 `TOOL_CALL` 节点，再加 7 类过程节点 | `attack_log_extractor.py`：`extract()` 对每条 trace 造 TOOL_CALL |
| **写图三 AG 路径 + 三 KG 路径，重叠易碎** | AG：`apply_planner_decision`/`apply_execution_result(_record_execution_result_in_ag)`/`apply_log_extraction`；KG：`StateWriter`/`_structured_stage_state_deltas`/`_fact_state_deltas` | `result_applier.py` |
| **双层 LLM 决策反复横跳** | planner 选 agent，5 个执行器又各自跑 max_steps 自判 finish/replan | 已删除的 registry/多执行器结构 |

---

## 2. 目标架构总览

```
                    ┌──────────────────────────────────────────────┐
                    │  常驻极小摘要（每轮注入）：节点计数、           │
                    │  eligible_for_stop、achieved_level、missing、 │
                    │  最近 3 个 attack step                         │
                    └──────────────────────────────────────────────┘
   ┌────────────┐        directive          ┌────────────────────┐
   │  Planner   │ ───────────────────────▶ │  ExecutionAgent    │
   │  (agentic) │                           │  (单目标一轮)       │
   │            │ ◀─────────────────────── │  内部可分析多工具    │
   └────────────┘        RoundResult        └────────────────────┘
        │ │  读工具: kg_query/ag_timeline/get_round_log              │ 过程落
        │ │  写工具: record_finding/record_attack_step/link_evidence │ Round Log 文件
        ▼ ▼                                                          ▼
   ┌─────────┐        ┌─────────┐        ┌──────────────┐    ┌──────────────┐
   │   KG    │        │   AG    │        │ SuccessTracker│   │  Round Log   │
   │ 状态真相 │◀──────│ 结果叙事 │───────▶│ (确定性成功门) │   │  过程明细     │
   └─────────┘  ADVANCED └────────┘ log_ref └──────────────┘   └──────────────┘
```

**三层职责分离**：

```
KG  = 世界状态 / 真相    "目标现在是什么样"       —— 实体+关系，去重、持久
AG  = 决策结果叙事        "我们做了什么、结果如何"  —— 每轮一个结果节点，串成时间线
Log = 执行过程明细        "这一轮具体怎么跑的"      —— 落文件，AG 只存 log_ref 指针
```

---

## 3. 图模型设计

### 3.1 KG —— 基本保留（环境状态真相）

KG 装"关于目标的事实"，按现有分层保留：

- 资产：`Host` `Service` `NetworkZone` `DataAsset`
- 身份/访问：`Identity` `Credential` `Session` `PrivilegeState` `PivotRoute`
- 漏洞/风险：`VulnerabilityCandidate` `Vulnerability` `ExploitCapability`
- 证据/发现：`Observation` `Evidence` `Finding`
- 目标：`Goal` `GoalCheck` `GoalProof`

主机上拿到的权限 = `Session` / `PrivilegeState` 连到 `Host`。
写入来源收敛为两条：① 工具 trace 自动抽取（确定性）；② planner 写工具的判断级 finding。
KG 是 planner 读工具的查询对象。**不大改**，仅删除纯过程性的节点类型（若有）。

### 3.2 AG —— 砍成结果时间线（2 类节点）

从 9 种过程节点降到 2 种结果节点：

| 节点 | 字段 |
|---|---|
| `ATTACK_STEP`（每轮 1 个） | `cycle_index`, `capability`, `objective`, `target_ref`, `status`(success/fail/blocked), `summary`, `evidence_refs`, `kg_node_refs`（本轮产出/用到的 KG 节点）, `log_ref`（→round log 文件） |
| `GOAL_OUTCOME`（终态） | `result`(success/fail), `achieved_level`, `proof_token_ref`, `redacted_summary` |

边只留两类：

- `NEXT`：step→step，时间线；replan/回溯时可分叉成多条路径，展示放弃的分支
- `ADVANCED`：step→它产出的 KG 节点，做 AG↔KG 交叉引用

**删除**：`TOOL_CALL` / `AGENT_EXECUTION` / `PLANNER_DECISION` / `HANDOFF_SUGGESTION` / `BLOCKED_REASON` 等过程节点——信息分别去往 KG（事实）、Log（过程）、或折叠成 `ATTACK_STEP` 字段。

**关键性质 —— AG 节点"作为节点粗、靠连边精确"**：`ATTACK_STEP` 本身只有几个字段，但通过 `kg_node_refs`/`ADVANCED` 指向具体 KG 节点、通过 `log_ref` 指向完整过程。细节没丢，只是放在了正确的层：

```
ATTACK_STEP #4: capability=exploit, target=host:web01, status=success,
  summary="利用 CVE-… 获取 www-data shell",
  kg_node_refs=[session:web01-www-data, privstate:web01-www-data, evidence:exploit-trace],
  log_ref=round-4.log
```

→ 既不笼统（指向精确事实），也不与 log 重复（AG 存"发生了什么推进+指针"，log 存"具体怎么跑"，零重叠）。

### 3.3 Log —— 新增第三层（过程明细）

ExecutionAgent 每轮的完整 ReAct 轨迹（工具调用 + stdout/stderr + 中间分析）写成一份 round log 文件（复用/扩展现有 `TxtTraceLogger`），`ATTACK_STEP.log_ref` 只存指针。**这是 AG 能"只记结果"的前提**——过程没丢，只是搬出了图。

---

## 4. 执行器：5 个 StageAgent → 1 个 ExecutionAgent

### 4.1 模型

ExecutionAgent 是**一个有进攻能力的真实渗透 agent**：拿到 planner 的一个目标（如"侦察 X" / "利用 Y 漏洞拿下 X 的立足点" / "从 X 横向到内网 Z"），可连续调用工具、**调用后自行分析输出、决定下一个工具**，直到完成该轮目标。它**真打**——生成并运行真实 exploit/payload、拿 shell、跑命令、提权、建会话、横向、读目标数据；不是"安全验证器"（见 §4.3）。

```python
class RoundDirective(BaseModel):     # planner → executor
    capability: Literal["recon","analysis","exploit","pivot","lateral","goal","evidence"]
    objective: str
    target_refs: list[GraphRef]
    allowed_tools: list[str]          # 来自 capability→tools 配置表
    tool_hints: list[dict] = []       # 建议工具序列（非强制）
    max_tools: int = 8                # 本轮工具调用上限
    success_hint: str | None = None   # 本轮"够了"的判断提示（不是任务成功）

class RoundResult(BaseModel):         # executor → planner
    cycle_index: int
    capability: str
    tool_traces: list[ToolTrace]
    extracted_facts: list[ExtractedFact]   # 自动确定性抽取
    raw_summary: str
    log_ref: str                       # round log 文件指针
    objective_met: bool                # 仅"本轮目标是否达成"，不碰任务成功
```

### 4.2 决策权重新切分（消除双层横跳）

| 判定 | executor | planner |
|---|---|---|
| 本轮目标是否达成、要不要再多调一个工具 | ✅ | ❌ |
| 整体任务是否成功 / 是否 stop / 下一轮做什么 | ❌ | ✅ |

executor **绝不**碰 `stop_success`/`replan`/选下一个 capability，只回答"我这一轮干完了，这是结果"。
capability 从"5 个 agent 类"降级为 directive 上的一个**标签** + 一张 `capability→allowed_tools` 配置表。
其中 `capability=exploit` 指**真实利用并获取访问**（拿 shell/会话），不是 `exploit_validation`（"证明漏洞存在但不打"）。
GoalAgent 的目标校验职责回归 Verify 阶段（那里本就跑 `SuccessConditionTracker`+`GoalOracle`）。

### 4.3 权限与渗透姿态（真实渗透，非验证 —— 一等公民）

框架定位是**自动化渗透测试**，执行器的默认姿态是**真打**，不是"安全验证"。这是 Phase H 已落的方向：
历史上的客户端 `PolicyEngine` 已删除；不再以未接线的客户端策略层声称阻断 active_exploit /
command_execution / file_write / reverse_callback / destructive，而是转成风险标签交给默认放行的
`RiskPolicy`（`block_*` 默认全 False）。**重构必须让执行器的 prompt/能力跟上引擎的放开**——
旧 `exploit_validation_agent.md` 那一串 "must not: generate exploit code / shell payloads / destructive /
reverse callbacks / write files / persist / bypass auth / brute force" **整体删除**。

**默认放开（授权 scope 内）**：生成并运行真实 exploit/payload、获取 shell/反弹、跑任意命令、写文件、
提权、持久化、暴力/凭证攻击、横向移动、读目标数据。`run_command` / `session_open` /
`pivot_route_probe` / `internal_service_discover` 等**默认可用**，不再 deny。

**权限边界只剩三条**（其余一律默认放行）：
1. **授权 scope 边界**：由 **mcp_lab 服务端 `authorized_hosts`** 强制（工具进程外校验目标，越界即拒）。客户端 executor 的 `policy_check` 仅 audit-only。（历史说明：客户端 `PolicyEngine.evaluate_target_scope` 从未接线到活路径，已删除。）
2. **secrets 不落地**：flag/token/marker 永不进 KG/AG/Log/审计/LLM；只在 GoalOracle 内做 HMAC 比对。
3. **可选 profile `safety:` 薄护栏**：默认全放行；只有 profile 显式重新打开某个 `block_*` 才生效。

**不要混淆两件事**：
- *真实攻击 vs 安全验证* = 执行器的**进攻姿态**。我们选真实攻击。
- *成功取证 vs 限制攻击* = 契约里的 `goal_proof`/`controlled_data_read_proof` 是 **secrets-不落地的产物**：
  agent **真的读到**数据，只是把 secret 做 HMAC 不写进图。**真实渗透与成功取证并存**，取证不等于"只许安全验证"。

新 `execution_agent.md` 的基调：你是授权范围内的真实攻击者，目标是**达成 capability**（拿下/打穿），
用一切可用工具，唯一红线是不越授权 scope、不把 secret 写进任何图/日志。

---

## 5. Planner：带图工具的 agent

planner 升级为 agentic，用 typed 工具读写图，而非把 KG/AG 压成上下文 dict。

> **"agentic" 指什么**：当前 planner 是*一次性、被喂饭*的——orchestrator 用 `TwoGraphContextBuilder`
> 把图压成一个固定 dict 喂给 LLM，拿回一个 `PlannerDecision`，只能看 builder 选的那个切面。
> agentic planner = *带工具、自主*的：在一次 `decide()` 里可按需调读工具下钻图、推理、调写工具落判断、
> 再吐下一轮 directive，对"看什么、记什么"有自主权。对外仍是"一轮一个 directive"。

### 5.1 工具清单

**读工具（完全放开）**
- `kg_query(node_type, filters)` / `kg_get_node(id)` / `kg_neighbors(id, edge_type)`
- `ag_get_timeline()` / `ag_get_step(id)`
- `get_round_log(step_id)`（按需拉某轮明细）

**写工具（受约束 typed，LLM 不手写节点 JSON）**
- `record_finding(host_ref, title, severity, summary, evidence_refs)`
- `record_attack_step(capability, target_ref, status, summary, evidence_refs, kg_node_refs)`
- `link_evidence(node_ref, evidence_ref)`

**决策出口（结构化返回，非工具）**
- 下一轮 `RoundDirective`，或 `stop_success` / `stop_failed` / `pause_for_review`

一次 planner agent 会话内可能多次工具往返，但**对外仍是"一轮一个 directive"**，主循环不变复杂。
保留一份**极小常驻摘要**随每轮注入（节点计数、`eligible_for_stop`、`achieved_level`、`missing`、最近 3 个 step），给 planner 定向，细节再用读工具下钻。

### 5.2 typed 工具原理

typed 工具 = 用 function calling 暴露写图能力，入参是**严格类型 schema**（pydantic），运行时校验。
LLM 只填**叶子语义字段**，碰不到结构。

```
LLM 提供（语义）:  { host_ref:"web01", title:"SSTI on /search",
                    severity:"high", summary:"...", evidence_refs:["ev-3"] }
工具填充（结构+安全）:
  finding_id = stable_hash(...)            # ID 工具生成
  node_type  = NodeType.FINDING
  edge: FINDING --APPLIES_TO_HOST--> web01 # 连边工具来连
  provenance = {cycle, round}
  sanitize(summary)                         # 按 secret 模式脱敏
  apply_patch_batch(...)                    # 逐条容错
```

分工依据：LLM 擅长语义判断、不擅长引用完整性/ID/去重；工具正相反。沿这条线切。
→ LLM 不可能写出格式错乱的图，因为它从不创作结构。

### 5.3 写图整体流程（每轮，谁做什么）

```
executor 回传 RoundResult(tool_traces, extracted_facts, log_ref)
        │
  第1层 自动确定性（无 LLM）：
      ToolTraceFactExtractor → KG 实体/关系 patch → apply_patch_batch
      ⇒ 机器事实（主机/服务/会话/凭证…），占写入 80%+
        │
  第2层 planner 判断级（LLM 调 typed 工具）：
      planner 推理 RoundResult + 读图 → record_finding / record_attack_step / link_evidence
      ⇒ 只有"判断"才产生的信息（结论、严重性、step 状态）
        │
   每个写工具调用：schema 校验 + 脱敏 + 逐条容错 + 与控制流解耦
```

LLM 只参与第 2 层、且只"决定记什么 + 供给语义内容"；结构/ID/连边/脱敏/持久化全由工具做；机器事实（第 1 层）完全绕过 LLM。

---

## 6. 写图与控制流解耦（写失败不卡流程）

> **图是派生的、咨询性的记录，永远不作为控制流闸门。**

下一步该做什么只由两样驱动：① 确定性成功追踪器读"图里现在有什么"；② planner 对手里的 `RoundResult` 的推理。
某次写图失败 → 记日志、继续；最坏是可视化里缺个节点，**能力不受影响**。
`apply_patch_batch` 已是逐条容错（一条坏 delta 不拖垮整批），保留此性质。
→ "硬编码写图写不成功是否导致无法进行下一步"：**不会，刻意设计如此**。这也是 typed 工具优于自由写图之处——部分成功、逐条容错。

---

## 7. 成功契约设计（多环境 / 多主机 / 看渗透程度 / 不违背黑盒）

### 7.1 原则：成功不由 LLM "感觉"判定

> 成功**声明式定义在每个环境的契约里**，**确定性评估**，planner 只**读判决**决定何时 stop，不自己判赢。

复用现有 `OperationProfile` + `SuccessContract` + `SuccessConditionTracker`，扩展"多标志 + 分级"。

### 7.2 黑盒：谁知道答案、谁不知道

| 角色 | 知道环境真相吗 | 拿到什么 |
|---|---|---|
| 出题人/评估器 | 知道（这是评分 rubric，合法） | 契约 + GoalOracle 里的私密 secret |
| GoalOracle | 持有 secret（env 变量） | 对 agent 提交物做 HMAC 比对，只回 pass/fail + 不透明 proof_token |
| agent（planner+executor） | **不知道** | 授权 scope 边界 + 自然语言任务目标 + 成功判决（eligible_for_stop / 缺哪类证据） |

### 7.3 契约条件编写规范（用"成功的形状"，不用"答案"）

```yaml
# ✅ 黑盒安全：抽象可验证谓词，对照 agent 自己建的 KG 求值
require_all:
  - foothold_internal      # 内网区域存在一个已验证会话
  - goal_proof_valid       # 存在针对任务目标的有效 GoalProof
require_chain:
  - [foothold_entry, foothold_internal]    # 必须经授权路径，不能跳关
  - [foothold_internal, goal_proof_valid]
condition_bindings:
  foothold_internal:
    predicate: exists_node
    args: { graph: kg, type: SESSION, filters: { zone_ref: "internal" } }
  goal_proof_valid:
    predicate: oracle_proof_valid
    args: { goal_id: "mission-goal" }       # 真值在 oracle，契约里没有 flag
```

```yaml
# ❌ 泄漏黑盒（禁止）
require_all:
  - "host 10.0.0.5 has CVE-2017-5638"        # 暴露拓扑+漏洞
  - "flag equals FLAG{...}"                    # 暴露答案
  - "exploit struts2 on port 8080"            # 暴露技术路线
```

四条规则：

1. **条件用抽象可验证谓词**：描述成功的*形状*（"存在经授权路径到达的、类别 C 的证明"），不描述路径或目标身份。
2. **答案物料（flag/token/marker）只在 GoalOracle**，永不进契约表面；agent 必须真的发现并提交，oracle 用 env secret 做 HMAC 校验，只回 pass/fail（= 现有 secrets 不变量）。
3. **谓词对照 agent 自己发现建起来的 KG**：`exists_node(SESSION, zone=internal)` 只有 agent 真的拿到会话才为真——成功 grounding 在实际发现上。zone/CIDR 只是授权边界（合法授权测试里 agent 知道 scope 是正当的）。
4. **`missing` 反馈只到"目标类别"粒度**：契约条件名和 `missing` 类别必须仅凭自然语言任务简报就能推得出来。
   - ✅ `missing: [internal_db_read_proof]`
   - ❌ `missing: ["need to exploit CVE-X on web01"]`

### 7.4 分级里程碑（看渗透程度）+ 多主机量化

```yaml
levels:
  minimal:  [foothold_any_host]                        # 拿到任一立足点
  standard: [foothold_any_host, pivot_to_internal]     # + 打进内网
  full:     [foothold_min_hosts, internal_goal_proof]  # + 多主机 + 目标证明
condition_bindings:
  foothold_min_hosts:
    predicate: count_nodes_at_least                     # 多主机：量化、不点名
    args: { graph: kg, type: SESSION, filters: { zone_ref: "internal" }, min_count: 2 }
```

追踪器返回 `achieved_level`（当前达到哪档）+ 每档 `missing`。
操作可配置"打到 standard 就停"，或"尽力打到最深、报告打到哪层"。
多主机 = 抽象量化谓词（"≥N 台不同主机拿到立足点"），可度量但不点名，agent 自己发现有哪些主机。

---

## 8. 轮-条件映射 + 重试预算（防 thrashing）

旧架构反复来自两处，分别处理：

- *双层循环*（executor 自判 finish/replan）→ 已由"判定权全归 planner"消除。
- *轮目标没切好* → 三条约束兜底：
  1. **轮目标锚定契约 `missing` 条件**：planner 把每个缺失条件映射成一个 capability 轮，轮数 ≈ 契约条件数，非无界重试。
  2. **每条件配重试预算**（如同一 missing 最多 2 次尝试）；用尽后 planner 必须换路径或 `stop_failed`。
  3. **失败轮 status=fail 写进 AG，planner 看得到**，不重发相同 directive。

> **AG step 粒度自动对齐契约条件粒度**：契约把"服务发现"和"拿到立足点"列成两条独立条件 → planner 自然分两轮 → 两个 AG step；契约只关心"立足点" → 可能一轮。**想观察多细，就把契约写多细。**

---

## 9. 主循环伪码

```python
def run_operation(operation_id, goal):
    while True:
        # 刷新确定性成功门（供 planner 读）
        progress = success_tracker.evaluate(profile, contract, kg, ag, runtime)
        summary  = build_min_summary(kg, ag, progress)   # 极小常驻摘要

        # PLAN —— planner agent：读工具下钻 + 判断 + 决策（唯一 LLM 决策点）
        outcome = planner.decide(goal, summary, graph_read_tools, graph_write_tools)
        if outcome.action in {"stop_success", "stop_failed", "pause_for_review"}:
            finalize(outcome); break
        directive: RoundDirective = outcome.directive
        if retry_budget_exhausted(directive): planner_must_replan_or_stop(); continue

        # EXECUTE —— 单执行器跑完一个目标，过程落 log
        round_result: RoundResult = executor.run(directive)   # 内部可分析多工具

        # VERIFY + WRITE
        kg.apply(round_result.extracted_facts)                # 第1层：确定性机器事实
        # 第2层：planner 已在 decide() 内通过 typed 工具写 finding / attack_step
        # （每次写：校验+脱敏+逐条容错+与控制流解耦）
        save_round_log(round_result.log_ref)
    return run_summary(operation_id)
```

---

## 10. 删 / 改 / 增 清单

**删除**
- `src/core/runtime/attack_log_extractor.py`
- `src/core/execution/agents/{recon,vuln_analysis,exploit_validation,access_pivot,goal}_agent.py`
- `src/core/execution/advisors/*`（5 个）
- `src/core/execution/context/*`（5 个 builder，合成 1 个或内联）
- `src/core/execution/prompts/{recon,vuln_analysis,exploit_validation,access_pivot,goal}_agent.md`（含其中 `exploit_validation_agent.md` 那一串 "must not …" 验证姿态约束，整体废弃）
- `result_applier.apply_log_extraction`、`_record_execution_result_in_ag` 的 TOOL_CALL 循环

**修改**
- `src/core/models/ag.py` + `attack_process.py`：节点类型砍到 `ATTACK_STEP`/`GOAL_OUTCOME`，边砍到 `NEXT`/`ADVANCED`
- `src/core/runtime/result_applier.py`：删两条多余 KG 路径（保留 `ToolTraceFactExtractor` 为唯一机器事实源），AG 写改为单 step 节点
- `src/core/execution/{registry,dispatcher}.py`：退化为单执行器入口（或删除）
- `src/core/execution/llm_stage_advisor.py`：去掉 `PROMPT_BY_AGENT` 5 路分支
- `src/core/planning/mission_planner_agent.py` + `llm_mission_planner_advisor.py`：planner 改为带图工具的 agent，输出 `RoundDirective` / stop 决策
- `src/core/planning/prompts/planner_global_control.md`：改为驱动图工具 + 输出 directive + 自判 stop（读 `eligible_for_stop`/`achieved_level`）
- `src/app/orchestrator.py`：`run_operation_cycle` 收敛为第 9 节单循环
- `src/core/agents/graph_context.py`：从"全量压缩"改为"极小常驻摘要"

**新增**
- `src/core/execution/execution_agent.py`：单 ExecutionAgent（单目标一轮，内部有界循环，不判 stop/success）
- `RoundDirective` / `RoundResult` 模型
- `src/core/execution/prompts/execution_agent.md`
- `capability → allowed_tools` 配置表
- planner 图工具：`kg_query`/`kg_get_node`/`kg_neighbors`/`ag_get_timeline`/`ag_get_step`/`get_round_log` + `record_finding`/`record_attack_step`/`link_evidence`
- 成功契约 `levels` / `achieved_level` 支持（`SuccessConditionTracker` 扩展）

**保留不动**
- KG 模型主体、`ToolTraceFactExtractor`（唯一机器事实源）
- `SuccessConditionTracker` / `PredicateEngine` / `GoalOracle`（确定性成功门）
- 安全不变量：mcp_lab 服务端 `authorized_hosts` 授权 scope 边界、secrets 永不进图

---

## 11. 分阶段落地（每阶段独立可验证）

| 阶段 | 内容 | 验证 |
|---|---|---|
| **P1 图瘦身 + 写图收敛** | AG 砍 2 节点 2 边；删 `attack_log_extractor` 与 `apply_log_extraction`；KG 收敛到 `ToolTraceFactExtractor` 单路径；AG 写改单 step | 现有 e2e run 仍跑通，AG 节点数大幅下降 |
| **P2 单执行器** | `ExecutionAgent` + `RoundDirective`/`RoundResult` + capability→tools 表；旧 5 agent 暂经适配层桥接 | 单 capability 轮在 full_chain_lab 跑通 |
| **P3 主循环 + agentic planner** | 主循环改第 9 节；planner 接图读写工具；极小常驻摘要 | 端到端走完整 PEV 循环 |
| **P4 成功契约分级 + 清理** | `levels`/`achieved_level`；删旧 agent/advisor/context/prompt；清理测试 | 多主机分级契约 e2e + 测试全绿 |

---

## 12. 渗透姿态与不变量（重构后仍须成立）

**默认姿态：真实渗透**。授权 scope 内默认放开真实利用/命令/会话/文件写/反弹/提权/持久化/暴力/横向/读数据；
框架代码不得再硬编码攻击限制。执行器是真实攻击者，不是安全验证器（见 §4.3）。

在此姿态下，仅保留五条不变量：

1. **授权 scope 边界**：由 **mcp_lab 服务端 `authorized_hosts`** 强制（工具进程外校验目标，越界即拒）。客户端 `policy_check` 仅 audit-only。（客户端 `PolicyEngine` 硬闸从未接线，已删除。）
2. **密钥不落地**：flag/token/marker 永不进 KG/AG/Log/审计/LLM 上下文；只在 GoalOracle 内与 env secret 做 HMAC 比对。
3. **成功权威在契约**：`eligible_for_stop`/`achieved_level` 由确定性追踪器计算，planner 只读不改。
4. **写图非控制闸门**：任何写图失败都不得阻断下一步。
5. **黑盒完整性**：契约只暴露"成功的形状"与"目标类别"，不暴露答案/拓扑/技术路线。
