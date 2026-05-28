# Aegra LLM Stage Architecture

```text
Graph State Layer
KG / AG / TG / Runtime / Policy
        ↓
Planner Agent
Global Goal -> Stage Tasks
        ↓
Stage Agents
ReconAgent / VulnAnalysisAgent / ExploitAgent / AccessPivotAgent / GoalAgent
        ↓
MCP Tool Layer
recon / vuln / exploit / access / credential / pivot / goal tools
        ↓
StageResult / ToolTrace
        ↓
ResultApplier.apply()
        ↓
Runtime / KG / AG / TG Updates
        ↓
Next Loop
```

The Planner Agent is an LLM Agent. It owns goal understanding, GraphStateSnapshot analysis, stage task decomposition, TG update intent generation and next StageTask selection. Code must not replace that with keyword rules or a fixed recon -> vuln -> exploit -> pivot -> goal sequence.

Stage Agents are also LLM Agents. ReconAgent, VulnAnalysisAgent, ExploitAgent, AccessPivotAgent and GoalAgent own their full stage decisions: whether the task is ready, which MCP tools to call, arguments, rationale, result interpretation, completion status, retry, replan, handoff or stop. Code only builds prompts, supplies GraphStateSnapshot and Policy, validates JSON, enforces policy before tool calls, records ToolTrace and returns StageResult.

MCP tools provide capability only. They return raw output, parsed fields, errors and evidence references. They do not write KG, AG, TG or Runtime.

ResultApplier.apply() is the only graph state write boundary. PlannerResult and StageResult carry graph_update_intents, facts, evidence, task proposals and runtime hints. ResultApplier performs deterministic validation, merge, status updates, provenance, audit logging and confidence-aware writeback. It does not call the LLM for core judgment.

Policy is read by Planner Agent, Stage Agents and enforced before MCP calls. Sensitive exploit, access, credential and pivot tools require explicit authorization context. Denied calls produce blocked ToolTrace/StageResult and are still applied so Runtime and TG record the blocker.
