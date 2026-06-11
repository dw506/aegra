"""Real black-box e2e driver to confirm tool results reach the KG.

Drives the genuine runtime link: live LLM planner + in-process MCP lab tools +
the real PhaseTwoResultApplier.apply_stage_result write-graph path. After the run
it dumps KG node/edge type counts and the per-cycle GRAPH_WRITE diagnostics so we
can see whether tool-call results actually landed in the knowledge graph.
"""

from __future__ import annotations

import collections
import json
import os

from _repo_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.core.agents.packy_llm import load_llm_env_file


def _configure_env() -> None:
    load_llm_env_file(".env")
    # 只接进程内 lab stdio server，避开 docker/http 那些不可达的 server。
    os.environ.setdefault("AEGRA_MCP_ENABLED", "1")
    os.environ.setdefault("AEGRA_MCP_FIRST", "1")
    os.environ.setdefault("AEGRA_LAB_MODE", "1")
    os.environ["AEGRA_MCP_CONFIG_JSON"] = json.dumps(
        {
            "servers": {
                "pentest-tools": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "src.integrations.mcp_lab.server"],
                    "cwd": os.getcwd(),
                    "env": {"AEGRA_LAB_MODE": "1"},
                }
            }
        }
    )


def main() -> None:
    _configure_env()

    from src.app.orchestrator import AppOrchestrator, TargetHost
    from src.app.settings import AppSettings
    from src.core.agents.agent_protocol import GraphRef, GraphScope

    base_url = os.getenv("AEGRA_VULHUB_BASE_URL", "http://127.0.0.1:8080/")
    host = "127.0.0.1"
    port = 8080
    op_id = "blackbox-writegraph-e2e"

    settings = AppSettings.from_env().model_copy(
        update={
            "runtime_store_backend": "memory",
            "runtime_policy": {"authorized_hosts": [host]},
        }
    )
    orch = AppOrchestrator(settings=settings)
    print(f"mcp_client configured: {orch.mcp_client is not None}")

    orch.create_operation(op_id)
    orch.import_targets(
        op_id,
        [
            TargetHost(
                kind="url",
                value=base_url,
                address=host,
                hostname=None,
                port=port,
                protocol="tcp",
                url=base_url,
                tags=["docker-lab", "authorized"],
            )
        ],
    )
    orch.start_operation(op_id)

    graph_refs = [
        GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
        GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph"),
    ]
    planner_payload = {
        "mission_goal": (
            f"Authorized black-box assessment of the Docker lab HTTP endpoint {base_url}. "
            "Discover the host and exposed service, then record findings."
        ),
        "policy_context": {
            "authorized": True,
            "authorized_hosts": [host],
            "authorized_urls": [base_url],
        },
    }

    results = orch.run_until_quiescent(
        op_id,
        graph_refs=graph_refs,
        planner_payload=planner_payload,
        max_cycles=int(os.getenv("BLACKBOX_MAX_CYCLES", "4")),
    )

    print(f"\n=== ran {len(results)} cycle(s) ===")
    for idx, cycle in enumerate(results):
        for apply_idx, ap in enumerate(cycle.apply_results or []):
            diag = getattr(ap, "kg_write_diagnostics", None)
            deltas = len(getattr(ap, "kg_state_deltas", []) or [])
            if diag or deltas:
                print(f"cycle[{idx}].apply[{apply_idx}]: kg_deltas={deltas} diagnostics={diag}")

    kg = orch.graph_memory_store.load_kg(op_id)
    node_types = collections.Counter(node.type.value for node in kg.list_nodes())
    edge_types = collections.Counter(edge.type.value for edge in kg.list_edges())
    print("\n=== FINAL KG ===")
    print("node types:", dict(node_types))
    print("edge types:", dict(edge_types))
    print("total nodes:", sum(node_types.values()), "| total edges:", sum(edge_types.values()))

    tool_evidence = [
        n for n in kg.list_nodes(type="Evidence")
        if (n.properties or {}).get("evidence_kind") == "tool_result"
        or (n.properties or {}).get("tool_name")
    ]
    print("tool-result evidence nodes:", len(tool_evidence))

    wrote_tool_facts = bool(node_types.get("Host") or node_types.get("Service") or tool_evidence)
    print("\nRESULT:", "PASS - tool results reached KG" if wrote_tool_facts else "FAIL - no tool-derived facts in KG")


if __name__ == "__main__":
    main()
