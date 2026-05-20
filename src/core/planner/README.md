Core Planning Kernel
====================

This package contains deterministic planning and critic kernels.

Rules:

- Do not import `src.core.agents.*`.
- Do not import LLM clients or advisor implementations.
- Do not import runtime stores or mutate runtime state.
- Accept AG/TG/model inputs and plain configuration only.
- Return kernel models such as `PlanningResult`, `KernelCriticResult`, and
  `TaskKernelCriticResult`.

Agent wrappers in `src.core.agents` may import this package and translate kernel
results into `AgentOutput` records. The import direction is intentionally one-way:

`agents -> planner kernel`
