# Aegra 架构重构设计：从"验证 harness"到"自主渗透系统"

> 目的：把 Aegra 从"贴着渗透标签的受控验证框架"重构为真正的**自主渗透系统**，
> 同时保留一条**薄合法性边界**。本文记录方向判断、目标架构与迁移顺序。

---

## 0. 问题诊断（为什么"好模型跑出差结果"）

系统的运行模型仍是验证 harness：

- **planner 不是 agent**：`LLMMissionPlannerAdvisor.propose_next_decision` 是 `context → 一次 complete_chat → 一个决策`，无 tool-use 循环、无自适应查图、无多步迭代。真正掌控节奏的是确定性代码（`run_operation_cycle` / `run_until_quiescent`），LLM 只是循环里被调用一次的"选 capability + 写 objective"单发预言机。
- **目标 = 验证里程碑**：planner 的每轮 objective 从成功合约的 `missing` 条件派生，好模型只做"刚好勾掉下一条件"的最小动作，而非真实纵深突破。
- **工具焊死在 lab**：工具直接 emit 合约条件名（`tools.py` 里 `{"condition": "restricted_data_service_discovered"}`、`post_access_observable`），tool ↔ contract ↔ lab 三者耦合，换目标无一可迁移。
- **利用是罐头**：`exploit_profiles/*.yml` per-CVE 写死 payload，无 metasploit/sqlmap/真实后渗透 shell。
- **每轮 `max_tools=8` 硬上限 + 一轮一个 capability + executor 每轮交还 planner**：把流动的攻击切成僵硬阶段，拿到立足点也无法顺势深入。
- **session 是逻辑节点，不是活 shell**：无法跨轮在立足点上累积推进。

**结论**：智能被铺在轨道上——模型从未被给予 agency，只在刚性状态机里填槽，对着只为一个 lab 工作的工具。模型质量传导不到结果。

---

## 1. 核心原则：把两个"图"分开

当前混乱的根源是把两种完全不同的图当成一回事：

| | 控制流图 | 知识图（世界模型） |
|---|---|---|
| 是什么 | 步骤/状态机：节点=步骤，边=转移（LangGraph 那种） | KG/AG：engagement 状态——主机/服务/漏洞/会话/攻击路径 |
| 管什么 | **怎么跑**（HOW） | **知道什么**（WHAT） |
| 现状问题 | 被硬编码进 orchestrator（轨道） | planner 几乎不查它（单发，不自适应） |

**定调：控制流 agent 驱动，世界模型图驱动。agent 开车，图记忆。**

---

## 2. 图驱动还是 agent 驱动？图的作用？

**结合，但角色分清：**

- **控制流 = agent 驱动**。LLM 在循环里自己决定下一步。**绝不**把 recon→exploit→pivot 做成控制流图节点——那等于把"轨道"重新焊一遍。capability 阶段不该是图节点。
- **知识 = 图驱动**。KG/AG 留作图，但**重新定义角色**，只承担四件事：
  1. **世界模型 / 长期记忆**——agent 自适应查询和更新的状态
  2. **planner↔executor 共享黑板**（也是 checkpoint/resume 载体）
  3. **评估基底**——成功合约 / 影响谓词**只读图状态**，与工具解耦
  4. **审计叙事**——AG 是人类可读的攻击故事

  图**不是**控制流，**不是**阶段脚本。

---

## 3. LangChain / LangGraph 取舍

- **LangChain：不要**。一堆会漏自己意见的抽象（Chain/AgentExecutor 偏 legacy），已有领域模型 + MCP 工具层，它只会加间接层和债。LLM 调用直接用现有 `PackyLLMClient`/Provider SDK，工具走 MCP。

- **LangGraph：合理选择，但有前提**。它的价值**不是**"让它变 agentic"（裸 Python 也能写 tool-use 循环），而是：①单一类型化共享 state 贯穿节点（正好治第 5 节数据转换病）、②显式可推理控制流、③内置 checkpoint/resume/interrupt/human-in-the-loop、④tracing 生态。一个会 pivot、会中断续跑、可能并行子 agent 的渗透战役吃得下这些。
  - 用它时**只当"循环+状态"基底**，节点要薄（decide → act → observe → 路由回 decide），领域模型当 state 载荷，**不要**把 pentest 工作流编进图的硬边。

**强建议——别让"上 LangGraph"替代真正的活**：
> 先把架构改对（planner 真迭代 agent + 图角色归位 + 工具解耦），**再**决定要不要 LangGraph 当基底。循环写对了，塞进 LangGraph 是机械活；循环是错的，LangGraph 救不了。当**收尾的基底选择**，不是开场银弹。

executor 已有 bounded tool 循环，比 planner 更接近 agent → **planner 的 agent 化优先级 > executor 的 LangGraph 化**。

---

## 4. 数据结构转换满天飞 —— 是限制，但是症状

病根是**历史累积**：5-agent→1-executor、TG 删除、双循环坍缩……每次重构留下 adapter 在新旧形态间翻译。证据：`pipeline_results` 把 lean 结果包回老 `AgentPipeline` 形状只为喂 observer、`ExecutionResult↔RoundResult`、`StageResultAdapter`、`events/reducer`、刚删的 `tool_trace/tool_traces` 重复。

**两个真实 bug 本身就是转换 bug**：`tool_traces` 冗余、`zone_ref` vs `destination_zone` 错配。每次形态转换都是 bug 温床 + 认知负担 + 数据流遮蔽。

**修法**（和 agent+图 重构是同一件事）：
> 收敛到**少数 canonical 领域模型 + 图作为唯一状态真相**。数据流变简单：
> **agent 读图 → 决策 → 工具回原始事实 → 事实写图 → 循环**。
> 不再需要 `ExecutionResult→RoundResult→PipelineResult→StageResult` 这串信封。成功评估读图。信封越少越好。

LangGraph 的"单一共享 state"是此问题的一个答案，但**原则比库重要**——纪律性砍信封同样拿到收益。

---

## 5. 工具增强重构

与上面一致：**通用能力 + 回原始事实进图 + 不 emit 合约条件名 + 真工具**。

- 工具契约极简：`输入参数 → 原始事实（写图）`；不知道 lab、不知道合约。
- 现状：侦察层已是薄封装真二进制（`nmap`/`nuclei`/`whatweb`/`ffuf`，`shutil.which` 探测），**弱在利用层是自写罐头**。
- 补强（freeform toolset）：`sqlmap`（真 SQLi）、**metasploit msfrpcd → Meterpreter session**（一并解决"无真实后渗透 shell"）、`hydra/netexec`、通用 `pivot_exec(route_id, argv)` / `session_exec(session_id, argv)`。handler 是几十行胶水，不重写工具逻辑；也可直接挂第三方工具的 MCP server。
- **两套 toolset 并存**：`full_pentest`（罐头，确定性，CI/回归）+ `freeform`（真工具，能力评估），靠 `AEGRA_MCP_TOOLSET` 切换。

---

## 6. 合法性边界：薄边界，别织进全身

边界要留，但收成一条**薄边**，不要把"验证姿态"织进 agent 推理/工具/目标的全身。合法性只集中在三处**边缘**：

- **scope 闸**（`authorized_hosts/CIDR`，服务端，命令在哪跑/能到哪）——唯一不可绕过
- **靶场隔离**（一次性容器、无外网出口、无真实系统路径）
- **终局证明 HMAC**（secrets-never-in-graph，KG/AG 不存字面 flag）

边界之内，**一切都是进攻的**。现在的毛病不是"有边界"，是边界糊到了全身。

---

## 7. 目标架构

```
        ┌─────────────────────────── 薄合法性边缘 ───────────────────────────┐
        │  scope闸 · 靶场隔离 · 终局HMAC                                      │
        │                                                                    │
        │   ┌── Planner Agent (真 tool-use 循环) ──┐                          │
        │   │  读图→战役决策→派 objective→观察→迭代 │                          │
        │   └──────────────┬─────────────────────┘                          │
        │                  │ 共享黑板                                         │
        │            ┌─────▼──────  KG/AG 世界模型  ──────┐  ◄── 成功评估只读图 │
        │            │  主机/服务/漏洞/会话/攻击路径/叙事    │                   │
        │            └─────▲──────────────────────────────┘                  │
        │                  │ 原始事实                                         │
        │   ┌── Execution Agent (tool-use 循环) ──┐                          │
        │   │  自主枚举/利用/提权/横向(不每步交还)   │                          │
        │   └──────────────┬─────────────────────┘                          │
        │          通用工具 (薄适配真工具: nmap/nuclei/msf/sqlmap/exec)        │
        └────────────────────────────────────────────────────────────────────┘
```

- **驱动**：agent（planner + executor 都是真循环）
- **图**：世界模型 / 黑板 / 评估基底 / 叙事——**不是控制流**
- **数据流**：读图→决策→工具回原始事实→写图，信封收敛到 ToolFact + 图 deltas

---

## 8. 迁移顺序（每步独立可验，风险递增）

### 8.0 生成这个顺序的一条原则

次序不是拍脑袋，是三条约束叠加后**唯一**能推出的：

1. **依赖方向（DAG）**：agent（控制流）是数据层的**消费者**——读世界模型图、把事实写回图、objective 从图状态派生。故「数据契约层」在 DAG 上游，「框架/agent」在下游。**上游先清。**
2. **减法先于加法**：删耦合/死代码是减法，**删不出错误的东西**（只移除确定是错的，不预设未来设计），可放心前置；定义新形态是加法/设计，**必须等约束它的东西（框架 + 真工具）到位**才做，否则返工。
3. **风险递增、每步独立可验**：低风险地基先钉死，不与高风险大改叠在一起调试。

套到这个已耦死的 brownfield 系统，就得到：**先自底向上剥耦合（减法）→ 再建框架（agent）→ 最后才定脚手架与依赖框架的设计（加法）**。
> 注意：这不是「先改数据判定、后定框架」。前置的是**拆耦合的减法**，不是数据设计。框架不能先定，因为框架的正确性依赖它跑在一个**已解耦**的数据层上——在耦合的数据上建 agent = 建完再拆（agent 会继承 §0 的 lab 轨道）。

### 8.1 六步（每步含「为什么在这」）

1. **解耦 / 成功评估去耦（= 附录 B 的 1-A）** — **[已完成 2026-06-28]**
   - 做：工具不再 emit 合约条件名，成功评估全移到 `SuccessConditionTracker` 纯读图，无 bindings 直接报错。
   - 为什么第一：它是 DAG 的**根**——无上游依赖，却被所有下游依赖（planner objective 从合约 missing 派生，§0；合约若还认识 lab 里程碑名，agent 化必继承 lab 轨道）。纯减法、局部、不依赖新框架、可独立回归。

2. **收信封 + 事实词汇通用化（吸收原 1-B）** ← 下一步
   - 做（减法为主）：砍 `pipeline_results` 及 `ExecutionResult→RoundResult→PipelineResult→StageResult` 信封串（§4 病根）；核心 `ToolTraceFactExtractor` 停止按 lab 工具名分发、停造 `LabFlag/LabHint`。
   - 做（最小加法）：定 canonical 模型（`ToolFact` / 图 delta / 一个 `PlannerDecision`）；extractor 改按工具声明的 `fact_kind` 分发；`LabFlag/LabHint`→通用 `Evidence/Proof`(带 `kind`)，合约 binding 同步。
   - 为什么在 1 之后 / 3 之前：成功评估去耦后数据流才收敛成「读图→决策→工具回事实→写图」，canonical 形状才清晰可定；而 agent（Step 3）操作的就是这套数据，得先定干净。
   - **验收约束**（设计时必须满足，非现在设计）：核心不认识 lab 工具名/节点类型；Host→Session→PivotRoute 因果边可表达（多主机成功判定的前置）。

3. **planner agent 化**
   - 做：`propose_next_decision` 单发预言机 → 真 tool-use 循环（读图工具进循环 + 多步迭代）；清 `advisor` 化石命名与死接口（附录 A）。
   - 为什么在 1/2 之后：第一个「真框架」改动，agent 一上来就读世界模型图、按图决策；图若还耦 lab（1/2 没做），agent 继承污染的世界模型与 lab 轨道 = 建完再拆。**这正是「不能先定框架」的核心。**

4. **executor 放开**
   - 做：去掉每轮交还 planner、放开 `max_tools` 硬上限、拿到立足点自主续打（§0 的「攻击被切成僵硬阶段」）。
   - 为什么在 3 之后：executor 已比 planner 更接近 agent（§3），但放开它的前提是 planner 已是真 agent（能消化自主推进后的图状态变化）。「planner agent 化优先级 > executor」。

5. **工具真能力（freeform toolset）**
   - 做：`sqlmap`（真 SQLi）、`msfrpcd→Meterpreter session`（解决「无真实后渗透 shell」）、`hydra/netexec`、通用 `pivot_exec/session_exec`；与罐头 `full_pentest` 并存，`AEGRA_MCP_TOOLSET` 切换。
   - 为什么在这：真工具产**真因果事实**（session 来自哪次 exploit、dump 经哪条 route）。**这一步解锁「多主机成功判定」**——图里有了真因果边，objective-子图判定才有东西可读。

6. **（可选收尾）LangGraph 作基底**
   - 为什么最后：§60 定论——其价值是「单一共享 state + checkpoint/resume + tracing」，不是「让系统变 agentic」。循环写对了塞进去是机械活，循环错了它救不了。收尾的基底选择，不是开场银弹。

### 8.2 横跨多步的「延后设计」项（真依赖框架的加法，全部后置）

| 设计项 | 落点 | 为什么不能更早 |
|---|---|---|
| 完整通用事实词汇 schema | Step 2（受 Step 5 影响的部分继续后延）| 真工具会带来新种类事实 |
| **多主机成功合约形态**（objective-over-子图、变量绑定、因果路径判定）| **Step 5 之后** | 要读真因果边，图里得先有 |
| LangGraph 脚手架 | Step 6 | 循环定型后才是机械活 |

**一句话总览**：减耦合（Step 1-2，数据层，减法为主）→ 建框架（Step 3-4，agent 真循环）→ 接真工具（Step 5，产因果事实）→ 定脚手架与依赖框架的判定设计（Step 6 + 成功合约，加法）。

---

## 附录 A：`advisor` 残留（"硬代码+LLM 拼凑"的化石证据）

- 代码里 `advisor` = `LLMMissionPlannerAdvisor`，即 planner 委托决策的那一步 LLM 调用。链路：
  `MissionPlannerAgent.decide() → advisor.propose_next_decision() → 一次 complete_chat()`。
- **命名是化石**：旧架构里 LLM 是"给确定性 planner 当顾问/打分"的，故称 advisor；v3 坍缩后它成了 planner 的**唯一大脑**，但外壳/命名没改。
- **启用条件**：`stage_llm_client is not None`（来自 `.env` 的 `AEGRA_LLM_API_KEY/BASE_URL/MODEL`），**不是**那几个 enable 标志。无 client → planner 返回 `replan / planner_llm_unavailable`，无硬编码兜底。
- **死配置**：`.env` 的 `AEGRA_ENABLE_PLANNER_LLM_ADVISOR` / `..._RANK_LLM_ADVISOR` / `..._GRAPH_LLM_PLANNER_ADVISOR` / `..._CRITIC_LLM_ADVISOR` 四个键**全 src 无人读取**，是旧"多 advisor"架构（planner/rank/graph/critic）的残留；critic 已删，rank/graph advisor 类已不存在。可直接删。
- 真把 planner agent 化（迁移第 3 步）时，`MissionPlannerAdvisor` Protocol + `propose_next_decision` 单发接口 + 这些死标志，应一并清理——advisor 从"被调用一次的顾问"变成"驱动循环的 agent 本体"。

> 进度：`.env` 的四个死配置标志已删除（2026-06-27）。

---

## 附录 B：Step 1 落地分析 —— 给框架核心"去 lab 化"（成功评估去耦 + 事实词汇通用化）

> 背景：定调是**通用渗透框架**，不是为 full_chain_lab 专门改。故 Step 1（§8）的真正目标不是"让 full_chain 跑通"，而是**清除框架核心层的环境特异**。本附录是动手前的现状勘验，代码引用为 2026-06-27 快照。

### B.1 现状：干净底座已存在且在用，耦合路径是"已被取代的遗留"

- ✅ `SuccessConditionTracker.evaluate`（`success_condition_tracker.py:45`）**已纯读图**——对每个合约条件按 `binding.predicate` 走 `PredicateEngine`，ctx 装 KG/AG/runtime 快照，**不读**工具的 `satisfied_conditions` 提示。
- ✅ `ToolTraceFactExtractor`（`tool_trace_fact_extractor.py`）已是"工具 trace → 类型化图事实"的桥（nmap/http/exploit/session/pivot/credential/goal_check…）。
- ✅ predicate 集本身通用：`exists_node`/`count_nodes_at_least`/`exists_edge`/`path_exists`/`chain_satisfied`… 由合约提供 args（节点类型/过滤器），引擎不认识具体环境。
- ✅ 活跃合约 `lab/environments/full_chain_lab/success_contract.yml` 的 13 个 `require_all` 条件**全部有 `condition_bindings`**，故主链路（`orchestrator.py:1391` 的 `if condition_bindings:` 分支）实际走的是干净读图路径。
- ⚠️ 决策点 `orchestrator._update_success_condition_progress:1391`：有 bindings → `SuccessConditionTracker`；**否则回退** `_inline_success_condition_progress`（耦合路径）。
- ⚠️ 旁证：工具 emit 的条件名（`restricted_data_service_discovered`）与合约里的条件名（`restricted_zone_service_discovered`）**根本对不上**——证明工具侧 emit + `_runtime_success_signals` 只喂 inline 回退，是遗留死路。

### B.2 核心层环境污染审计（真伪已核实）

| 位置 | 内容 | 判定 |
|---|---|---|
| `orchestrator.py:1483-1563` | `_runtime_success_signals`/`_inline_*` 硬编码 `dmz_service_discovered`/`database_file_read`/`goal_check_recorded` + `"dmz" in text` 文本启发式 | 🔴 真污染（核心编排认识 lab 里程碑名） |
| `tool_trace_fact_extractor.py:346,620` | 按 lab 工具名（`read_lab_marker`/`list_lab_hints`/`credential_discover_lab`）分发，造 `LabFlag`/`LabHint` 节点类型 | 🔴 真污染（核心抽取认识 lab 专用工具 + 专用节点类型） |
| `tools.py:1519,1621,1883,2066,2078` | 条件名 emit（`restricted_data_service_discovered`/`vulnerability_candidate_recorded`）+ `post_access_observable` | 🟡 集成层（mcp_lab 是罐头 toolset，本就偏 lab；但泄漏到框架合约的条件名要切断） |
| `execution_agent.py:430-431` `10.20.0.x` | 在**注释**里举例说明 token 匹配 bug 修复；逻辑读 `policy_context["blocked_hosts"]` | ✅ 误报（纯注释，逻辑通用） |
| `evaluation/models.py:157` `/flag` | 在 **docstring** 里描述"raw flag 永不入图"不变量 | ✅ 误报（文档，非硬编码） |

### B.3 框架/环境边界（定调）

```
框架核心 (src/app, src/core) —— 零环境知识
  提供：通用图 + 通用事实词汇(Host/Service/Vuln/Session/Credential/PivotRoute/Evidence)
        + 通用 predicate 引擎 + 按"通用能力语义"分发的 tool→fact 抽取
环境 (配置 + 隔离适配器) —— 所有特异性
  lab/*/success_contract.yml(用通用 predicate 表达的条件) + profile.yml
  + configs/*(vuln/exploit profile) + 具体 toolset(mcp_lab 是其一)
```

`success_contract.yml` 在 `lab/` 下、用通用 predicate 表达 —— 这是**正确的环境特异位置**；框架只提供评估机器，合约是每次交战的**输入**。这点架构本就对了，无需动。

### B.4 Step 1 = "给核心去 lab 化"，但**只保留 1-A**；1-B 移至 Step 2

**1-A 成功评估去耦**（核心 de-lab 最大块）—— **[已完成 2026-06-28]**
- ✅ 删 `orchestrator` 的 `_runtime_success_signals` / `_inline_success_condition_progress` / `_inline_level_results`（含所有硬编码 lab 条件名 + 文本启发式），连带删孤儿 `_achieved_level`(orchestrator 版) / `_dedupe_strings`(orchestrator 版)。
- ✅ `_update_success_condition_progress`：`if condition_bindings:` 改为**强制有 bindings**，无 bindings 直接 `raise ValueError`（不再静默走 inline）。
- ✅ `tools.py` 停止 emit 合约条件名（删 `satisfied_conditions`，原行 1519/1620-1621），连带删孤儿 `_is_restricted_data_service_name`；工具只回原始 `services/evidence`。
- ✅ 切 hint 桥：`execution_runtime_hints` 删消费方后已零读者，停写（`result_applier.py:98`）。
- ✅ 删"测被删行为"的死测试（`test_success_contract_loop_e2e.py` 整文件 + `test_operation_run_summary_contract.py` 两个 inline 用例）；`test_mcp_lab.py` 两个用例改为断言原始事实 + 加"不得再 emit 条件名"回归守卫。全量回归 **135 passed / 1 skipped**。
- **保留为原始事实**：`post_access_observable`（extractor:307 当 `ExploitCapability` 属性消费）；`goal_satisfied`（GoalOracle/HMAC 终局，§6 不动）。

**1-B 事实词汇通用化 —— 移至 Step 2（收信封）一起做。** 原 B.5 把 1-B 捆进 Step 1 是错的；1-B 含两件待遇不同的事：
- **(a) 减法**：核心 `ToolTraceFactExtractor` 停止按 lab 工具名分发（改为工具声明 `fact_kind: proof|credential|service|…`，extractor 按 kind 分发）、`LabFlag/LabHint` → 通用 `Evidence/Proof`(带 `kind`)。这是**解耦**，与 1-A 同病同根，本可现在做；但它**不卡任何下游**——1-A 已绿即证明（当前合约 binding 仍引用 `LabFlag/LabHint`，与 extractor 一致而自洽），故无须抢在 Step 1。
- **(b) 加法**：设计完整通用事实词汇（节点/边类型、因果边、proof 表达）。这是 **schema 设计**，与 Step 2 的 canonical `ToolFact / 图 delta` **本就是同一件事**；且真工具（Step 5：msf/sqlmap 产 meterpreter session / sql dump 等新种类事实）落地前词汇集不完整，现在定死必返工。
- 故 **(a)+(b) 在 Step 2 一次做**。验收约束：核心不许认识 lab 工具名/节点类型，且 Host→Session→PivotRoute 因果边要表达得出来（多主机成功判定的前置）。

### B.5 风险与决策（动手后回填）

- **先补后删（最大回归风险）**：删 `_runtime_success_signals` 前须确认其"将失去的条件满足来源"已由"predicate 绑定 + extractor 图事实"覆盖。→ **已由运行实证清账（见 B.6）**：inline 对活跃 full_chain 贡献为 0，删除是空操作；回归 135 passed 确认。
- **无 bindings 合约**：→ **已定：硬报错**（强制纪律），见 1-A。
- **GoalOracle/`goal_satisfied`**：§6 终局 HMAC 单独保留，本步未动。
- **范围（修正）**：原"1-A + 1-B 一起做"的建议**已作废**。**Step 1 = 1-A only（已完成）；1-B 整体移入 Step 2**——减法不卡下游、加法与 Step 2 canonical 模型重叠（理由见 B.4）。
- **不在本步**：收信封 + 1-B = Step 2；planner agent 化 = Step 3。

---

### B.6 运行实证：历史成功走的是 tracker 读图，不是工具自报

> 对 `lab/outputs/runtime/full-chain-*` 两次运行产物的核验（2026-06-28），把 B.1 的代码推断升级为运行实证。

- `runtime.json` 的 `condition_results` 全部带**真谓词名**（`exists_node`×24、`service_discovered_via_route`×2），`runtime_signal`（inline 指纹）出现 **0 次** → 判定 100% 走 `if condition_bindings:` 的 tracker 读图分支，inline `else` 从未触发。
- 双重佐证：工具 emit 的 `restricted_data_service_discovered` 与合约条件名 `restricted_zone_service_discovered` **字面对不上** → 工具自报既进不去 inline、名字也匹配不上，是双重失效的死路。
- 反面验证：gpt54 run `eligible_for_stop=false`、7 条 missing —— tracker 如实判不合格，**没被工具 emit 糊成成功**。
- 结论：**工具 emit 条件名从未伪造过成功判定**；它是死路 + 污染，1-A 已删之。无 bindings 合约才是真后门（会静默走 inline 自报），1-A 的"强制 bindings/报错"已堵死。

---

## 附录 C：字段「适配目标框架」判据 —— 三类死法 + ExecutionResult 收敛

> 背景：清理时发现「是否被引用/被消费」不足以判定去留。判据要升级为**「是否属于目标框架」**。Step 2 实操（2c + 死字段清理）暴露出三类，前两类已清，第三类是架构债。

### C.1 三类「该去」的字段（按隐蔽度递增）

1. **纯死**：从未赋值/读取（`ExtractedFact`(execution 版)、`graph_update_intents`/`GraphUpdateIntent`、`privilege_contexts`、`visual_summary`）。已删。
2. **假活/半成品**：被「消费」但终点是日志/测试/被忽略的上下文，无行为据它发生（`handoff_suggestion`/`next_capability_suggestion`/`next_capability_candidates`——planner 从不读，只进 `EXECUTION_FINISH` 日志）。已删。
3. **活但不适配**（最隐蔽）：真被消费、真写图，但属于**被目标框架淘汰的模式**。见 C.2。

### C.2 `ExecutionResult` 的「双写图通道」—— 第 3 类的核心

`ResultApplier` 把 `ExecutionResult` 写图时有**两条并行通道**：

- **通道①（目标正路）**：`tool_trace → ToolTraceFactExtractor → KG`，确定性、工具为事实权威。
- **通道②（待淘汰）**：LLM 在 finish payload 自述的 `observations/evidence/findings/discovered_entities/discovered_relations/capabilities_gained/credentials/sessions/pivot_routes` → ResultApplier 写 runtime+KG。

通道② **活且 load-bearing**（runtime 的 session/credential/pivot 目前唯一写入路径），但**不适配**：它让 executor 的 LLM 当「世界有哪些事实」的权威，与 §5「工具回原始事实」相悖，且与通道①双源竞争（§4 病）。

**迁移触发条件**：Step 5 真工具（msf session / pivot_exec / sqlmap）经 `tool_trace` 产出 session/pivot/credential + canonical ToolFact 落地后，通道①即可覆盖通道②，**届时整类移除，事实权威从 LLM 收归工具**。在此之前不动（会断 runtime 写入）。

### C.3 更进一步：`ExecutionResult` 本身大半是「当前框架脚手架」

数据形态仍显繁多的根因：**当前 executor 向 orchestrator 返回一份「消化过的胖报告」，planner 又被喂这份报告当上下文（`recent_execution_results`）——而非读图。** 目标框架（§7：planner 读图决策、executor 自主不每轮交还、图为共享黑板）一旦落地，这份胖报告大部分失去存在理由。按目标处置：

| 处置 | 字段 | 被哪步溶解 |
|---|---|---|
| **保留（薄）** | execution_id, status, summary, tool_trace（审计+事实源）| — |
| **移入图**（不再走 ExecutionResult）| 通道② 全部事实字段 + failed_hypotheses | Step 5（事实工具化）|
| **planner 改读图后失去理由** | 喂给 planner 的 `recent_execution_results` / `_compact_execution_result` 投影 | Step 3（planner 读图）|
| **executor 自主后失去理由** | 每轮交还的胖 payload 结构本身 | Step 4（executor 放开）|
| **可降级/删** | confidence, risk_level, created_at, retry_recommendation, policy_notes（多为 LLM 装饰，无决策消费）| 随上面收敛 |

**结论**：目标信封 = §4 的 `ToolFact + 图 delta + 一个 PlannerDecision/薄 round 状态`。当前 `ExecutionResult` 的胖结构是「executor 返回报告 + planner 读报告」模式的产物，**会被 Step 3/4/5 逐步溶解**，不是一次能删完——但要登记为债，避免将来又以「在用」为由保留。

---

## 附录 D：Step 3 落地 —— planner 单发 → 真 tool-use 循环（LangGraph-ready）

> 分支 `refactor/step3-planner-agent`。把 §0 诊断的「单发预言机」改成 LLM 自主迭代查图再决策的真 agent 循环。

### D.1 关键决策：手搓循环、但 LangGraph-ready；LangGraph 留到 Step 6
- 现在**不引入 LangGraph**（§60：循环写对了再上是机械活；现在上只给 planner 内层套、造混合体、拿不到系统级 resume——那要 planner+executor+外层一起，即 Step 6）。
- 但 Step 3 强制写成 **LangGraph 形状**，使 Step 6 仅「换驱动」、残留趋零。
- 现状铁证（要拆除的）：advisor prompt 明写「不能中途查图」；`tool_manifest()` 自述「push 模型、无 tool-call 循环、只暴露写工具」；`PlannerGraphTools` 原本无粒度读工具。

### D.2 三件套（substrate-agnostic + 薄驱动）
- **State**（`PlannerLoopState`）：operation/cycle/goal、seed_context（min_summary+progress 廉价种子）、`read_log`、`pending_call`、`decision`、`step/max_steps`。= 将来 LangGraph 的 State。
- **节点函数**（纯 `State→State`）：`decide`（构 prompt→一次 LLM→解析出「一个读工具调用」或「最终 PlannerOutcome」）、`act`（执行读工具、结果回灌 read_log）。
- **薄驱动**（~15 行 `for range(max_steps)`）：decide→有 final 即返；有 tool_call 则 act 再 decide；到顶兜底 replan。**Step 6 仅把这段换成 `StateGraph(...).invoke()` + checkpointer，节点/State 照搬。**
- 这段薄驱动是「只跑节点、零决策」的引擎——**不是 §0 批判的硬编码轨道**（决策权已在 LLM）。

### D.3 决策（已定）
- 循环机制 **L1**：手工 JSON（与 executor 一致、复用 `_extract_json_object`/`_fallback_outcome`），不用原生 function-calling（本仓 executor 亦未用）。
- 迭代上限 **max_steps=6**。
- 化石清理（折叠 `MissionPlannerAdvisor` Protocol + `propose_next_decision`，正名 advisor→agent 本体）放在**循环绿后**（3e）。

### D.4 分子步进度
- ✅ **3a 读工具**：`PlannerGraphTools` 增 `get_success_progress / query_kg_nodes / get_node / get_attack_steps / list_runtime` + `apply_read_call` 分发 + `read_tool_manifest`；`tool_manifest` 去掉「push 模型」自述、加 `read` 区。单测绿。
- ✅ **3b 循环**：新增 `planner_loop.py`（`PlannerLoopState` + `decide_node`/`act_node` + 薄驱动 `run_planner_loop`）。循环单测：读工具迭代→决策、budget 耗尽→replan、无 client→replan。
- ✅ **3c prompt**：删「不能中途查图」，改为「每轮返回 tool_call 或最终 PlannerOutcome」+ read_tools/read_log/read_budget。
- ✅ **3d 全量回归**：139 passed / 1 skipped。
- ✅ **3e 化石折叠（提前随改名一起做）**：见 D.6。

### D.5 范围边界
不动 executor（Step 4）、不接真工具（Step 5）、不做 ToolFact（Step 2 余项）。`recent_execution_results` 瘦身（C.3）本步并存不删。

### D.6 命名清理（附录 A 化石 + 缩短）
单发时代的 `advisor` 命名与多余 facade 全部收敛成**一个 `Planner` 类**（agent 本来就含 LLM，无需 `LLM*`/`*Advisor` 前后缀）：
- 删 `MissionPlannerAdvisor` Protocol + `propose_next_decision` 单发接口（附录 A 化石）。
- `LLMMissionPlannerAdvisor`(+Config) → `Planner`/`PlannerConfig`，文件 `llm_mission_planner_advisor.py` → `planner.py`。
- 删 facade `mission_planner_agent.py`：其 `decide`（建 state→跑循环→应用写工具→无 client 兜底）并入 `Planner.decide`。
- `planner_loop` 的循环角色 Protocol 命名 `PlannerTurn`（`run_turn`），驱动参数 `planner`。
- orchestrator/测试：`mission_planner` → `planner`。
- 结果：`Planner.decide`（orchestrator 入口）+ `Planner.run_turn`（每轮，被循环回调），一个类、短名。

---

## 附录 E：Step 4 设计 —— executor 放开 + planner/executor 功能定义（动手前讨论）

> 分支待建 `refactor/step4-executor`。本附录是动手前的设计定调，含 capability「误导」分析。

### E.1 两个 agent 的功能分工
一句话：**planner 管「打哪 + 何时停」（战略/全局），executor 管「怎么打透一轮」（战术/局部），图是两者间的真正通道。** 两者同形（`decide→act→observe` bounded 循环、LangGraph-ready）。

| | Planner | Execution Agent |
|---|---|---|
| 职能 | 读图→决定下一个有界目标 / 停/重规划/暂停；唯一全局控制者 | 把一个 directive 打透：开放工具空间自主选工具、经立足点操作、写事实回图 |
| 输入 | mission goal、图(读工具)、success_progress、policy、最近结果 | RoundDirective + 活立足点(sessions/routes) + 图读 + policy + 工具目录 |
| 输出 | `PlannerOutcome`{execute(RoundDirective)/replan/pause/stop_success/stop_failed} | status + summary + **原始事实经 tool_trace→extractor→图** + 新立足点回写 + 薄信号 |
| 独占 | 唯一发 stop_success/failed；全局节奏 | 工具选择+参数+经哪个立足点执行 |
| 禁止 | 出 shell/工具参数/固定阶段序列 | 决定全局停/战略、按固定阶段走、返回胖报告 |

契约：**向下给意图（RoundDirective），向上回事实（进图）+ 薄信号；真正状态交换在图里**（胖 ExecutionResult 随 C.3 溶解）。verify 独立：`SuccessConditionTracker` 读图算 `eligible_for_stop`，planner 只读它。

### E.2 执行 agent 的动作模型（不写死阶段）
两层自由度：
- **控制动作（固定、小）**：每步三动词 `call_mcp_tool`/`finish`/`need_replan`（循环骨架，Step 6 LangGraph 托管这层）。
- **工具选择（开放）**：`call_mcp_tool` 内自由从目录选任意 in-scope 工具 + 自定参数（agency 所在）。

**不要**把 recon/exploit/pivot 设成控制动词或硬闸（= §0 轨道）。它们是**软 capability**：planner 读图自适应排序（不定死流程），executor 在其聚焦内自由选工具。防「乱调」靠 **objective + scope闸 + budget**，不靠写死阶段。

### E.3 轮次限度 + 粒度
- **限度（停止判据）**：`success_hint` 达成→`finish`（语义停）；`max_tools` 硬顶 + 无进展守卫兜底。
- **粒度 = planner 的战略旋钮**：要不要「一轮 recon+exploit」由 planner 调 `objective` 范围决定（宽 objective + 够预算 = 一轮打透；窄 objective = 先看再决定）。confident 咬大口、risky 小步 verify。
- **「一轮 recon+exploit」靠宽 objective，不靠多选 capability**——executor 工具空间开放，objective 承载多动作。

### E.4 capability 的处置（误导分析）
- **误导源精确定位**：`execution_agent.py:647` 把 `capability=...` 喂进 executor 决策 prompt → 暗示「这是 exploit 轮」，可能带偏（跳过该做的侦察）。
- **修复（Step 4，聚焦）**：capability **移出 executor 决策上下文**；executor 只凭 `objective + success_hint` 推理。
- **保留用途**：capability 作 **AG ATTACK_STEP 粗标签 / 日志**（一轮一步需要个名字），可由 planner 出或事后从主导工具派生——但**不喂给 executor 决策**。
- **彻底删字段 = 后续大改**（织入 `CapabilityName` 类型、三个 model 字段、AG 标注、ResultApplier、~30 处、测试），随 C.3 的 `ExecutionResult` 瘦身一起收，**不在 Step 4**。

### E.5 多主机执行设计
- **target/route 是可选聚焦，非强绑**：侦察轮空 target、无 route 扫 zone；利用轮聚焦一主机。
- **经执行平面操作**：工具能 `session_exec(session_id)`/`pivot_exec(route_id)` 在已控主机/穿 pivot 执行（真活 shell 是 Step 5；Step 4 预留传输轴，用 `ExecutionRequest` 已有的 sessions/pivot_routes）。
- **图 = 共享立足点记忆**：executor 读活 session/route 集够到目标，新立足点写回图供复用。
- **一个 execution agent，顺序推进**：跨机纵深 = planner 一轮轮选目标 + 图累积立足点「涌现」；**不开 per-host 子 agent**（并行留 Step 6 LangGraph 的 state 隔离）。

### E.6 Step 4 范围
做：① 放开预算（max_tools 8→更大/可配、max_cycles 5→更大或按进度）② `success_hint` 成为真正停止判据 + 无进展守卫 ③ **capability 移出 executor 决策（仅留 AG 标签）** ④ executor 能经 session/route 操作（传输预留）⑤（可选）executor 整理成与 planner 对称的三件套（利于 Step 6）。
不做：真活 shell（Step 5）、并行子 agent（Step 6）、多主机成功判定（更后）、彻底删 capability 字段（随 C.3 收）。

### E.7 Step 4 落地状态（branch `refactor/step4-executor`）
- **① 放开预算（done）**：`RoundDirective.max_tools` 8→16、`ExecutionRequest.max_steps` 8→16（`execution/models.py`）；`run_until_quiescent max_cycles` 5→12（`orchestrator.py`）、API `max_cycles` default 5→12（`app/api/__init__.py`，`le=50` 不变）；planner 契约占位 + 提示词 `max_tools` 8→16，并加 rule 9：粒度由 objective 宽窄 + max_tools 调，不靠 chaining capability。
- **② success_hint 真停止判据 + 无进展守卫（done）**：executor system prompt 明确「从 objective+success_criteria 推理、达成即 finish、卡住即 need_replan」；`_ExecutionLoop.run` 加无进展守卫——连续 `NO_PROGRESS_LIMIT=3` 次「失败或重复」工具调用即提前停（`_tool_call_signature` 去掉 operation_id/trace_id 后比对）。
- **③ capability 移出 executor 决策（done）**：`_build_messages` context 删 `capability` 键；`required_context` 删死字段 `round_capability`。capability 仍在 request/result/日志中作 AG 标签，不入决策上下文。
- **④ session/route 传输预留（done）**：`_with_transport_context` 在工具调用前从 `request.sessions`/`pivot_routes` 取最近 active 的 `session_id`/`route_id` 盖进 call 参数（不覆盖已有值、不暴露给 LLM、记 `TRANSPORT_CONTEXT` 日志）；真活 shell 仍待 Step 5 消费。
- **⑤ 对称三件套（deferred）**：可选项，留待 Step 6 LangGraph 准备时做，避免本步引入大重构风险。
- 测试：新增 `test_no_progress_guard_stops_round_before_max_steps` / `test_capability_absent_from_executor_decision_context` / `test_active_session_transport_is_stamped_onto_tool_call`；全量 142 passed, 1 skipped。

---

## 附录 F：Step 5 设计 —— 纯真工具（去罐头）+ 数据结构收敛

> 分支 `refactor/step5-real-tools`。决策（与 user 定）：**不做"真/罐头并存 + toolset 切换"，直接收敛到单一真工具集**。本附录是动手前定调 + 分阶段落地记录。

### F.1 当前工具盘点（2026-06-29 快照，`mcp_lab/tools.py`）
- **已是真工具（留，零改）**：`run_command`/`nmap_scan`/`http_probe`/`web_fingerprint`/`web_discover`/`dns_lookup`/`tls_probe`/`tcp_connect_probe`/`http_basic_auth_check`/`nuclei_scan`/`whatweb_fingerprint`/`ffuf_discover`（真二进制/真发包/真 socket）。
- **罐头/模拟（待删，F3）**：`lab_authorized_exploit_execute`（profile-YAML 假利用）、`vuln_profile_match`（关键字判断，归 agent LLM）、`safe_vuln_validate`/`validation_precheck`（安全验证姿态，v3 已反转）、`session_open_lab`/`session_probe`（回显假 session）、`credential_check`（假凭证）、`identity_context_probe`/`privilege_context_probe`、`pivoted_nmap_scan`/`internal_service_discover`/`controlled_data_read_proof`（bespoke SSH 罐头）、`pivot_route_probe`/`pivot_route_register`、`post_access_observe`/`read_lab_marker`。
- **oracle/eval（留 lab 侧）**：`goal_check`/`internal_goal_check`/`chain_goal_check`/`success_condition_check`/`artifact_store`。

### F.2 关键事实：成功合约读图、不读工具名
`success_contract.yml` 用图谓词（`exists_node`/`service_discovered_via_route`）绑定 KG 节点类型，**不绑工具名**。故"换工具"只要真工具经 `ToolTraceFactExtractor` 产出相同 KG 节点类型（Service/ExploitCapability/Session/PivotRoute/Goal proof），合约层零改动——这让去罐头的牵连面可控。

### F.3 顺序铁律（C.2 警告）
通道② 是 runtime session/credential/pivot **当前唯一写入路径**。必须**先**让真工具经通道①（`tool_trace→ToolTraceFactExtractor→KG`）产出这些事实，**再**删通道② + 罐头，否则断 runtime 写入。**删除是收尾，不是开场。**

### F.4 分阶段
- **F1（done，本分支）**：真传输原语先行（不删任何东西，符合铁律）。
  - 新增 `pivot_exec(route_id, argv)`：包既有真 SSH egress `_run_via_configured_pivot`，跑任意 argv 穿 pivot；成功即因果证明 route 活 → 事实抽取器映射 `pivot_exec→_extract_pivot_route` 记 active PivotRoute。消费 Step 4 ④ 预留的 `route_id`。
  - 删死配置 `AEGRA_MCP_TOOLSET`（config + orchestrator 注释，已核实全代码无人读）——toolset 切换层取消（单一真工具集）。
  - 测试：`test_pivot_exec_runs_argv_through_configured_route` / `test_pivot_exec_accepts_shell_string_argv`；全量 144 passed, 1 skipped。
- **F2（todo，需 live 环境）**：接真引擎——`metasploit_exec`(msfrpcd→Meterpreter)/`sqlmap_scan`/`hydra_bruteforce`/`netexec_run` + `session_exec(session_id, argv)`。改 `ToolTraceFactExtractor` 解析真输出（msf session 元数据/sqlmap loot/hydra creds），抽取器 `parsed` 形态契约不变。**需 msfrpcd/二进制/靶机，不能离线盲写。**
- **F3（todo，F2 之后）**：删通道② + 罐头。`ExecutionResult` 删 `observations/evidence/findings/discovered_entities/discovered_relations/capabilities_gained/credentials/sessions/pivot_routes/failed_hypotheses` + 装饰字段 `confidence/risk_level/created_at/retry_recommendation/policy_notes`；`ResultApplier` 删通道② 写图/写 runtime 分支；删罐头工具 + `configs/{exploit,vuln}_profiles/*`；`execution_agent._normalize_tool_call_arguments` 删罐头工具名特例；修 B.1 命名 bug（`restricted_data_service_discovered`↔`restricted_zone_service_discovered`）。`ExecutionResult` 瘦成薄信封 `execution_id/status/summary/tool_trace`(+capability AG 标签)。
- **F4（todo）**：补全 ToolFact 通用词汇（真工具新事实种类）；多主机成功合约形态（objective-over-子图/变量绑定/因果路径判定，依赖 F2 的真因果边）。

### F.5 权衡：罐头兼作离线测试 fixture
罐头让套件无需真 msf/docker 离线确定性跑通。去罐头后：executor 层测试不受影响（`RecordingMCP`/`FakeStageLLM` mock）；`test_mcp_lab.py` 直测罐头实现的用例 F3 要重写；建议保留**最小 mock MCP server**（只回固定字节，无 profile 判断逻辑）供 CI，而非保留罐头业务逻辑。2 个 deferred option-C exploit-execute 红测随 F3 删罐头自然消失。
