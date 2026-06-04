import { AlertTriangle, CheckCircle2, Circle, Clock3 } from "lucide-react";
import type { GraphState } from "../graphState";
import {
  buildDisplayName,
  buildOperationOverview,
  buildStatePath,
  groupCycles,
  inferStepOrder,
  nodeType,
  readValue,
} from "../graphTransforms";
import type { OperationSummary, VisualNode } from "../types";

interface Props {
  ag: GraphState;
  runtime: GraphState;
  operation?: OperationSummary;
  selectedNode: VisualNode | null;
  onSelectNode: (node: VisualNode) => void;
}

export function AgTimelineView({ ag, runtime, operation, selectedNode, onSelectNode }: Props) {
  const overview = buildOperationOverview(operation, runtime, ag);
  const cycles = groupCycles(ag);
  const statePath = buildStatePath(ag);

  return (
    <div className="agTimelineView">
      <section className="operationOverview" aria-label="Operation overview">
        <Metric label="operation_id" value={overview.operationId || "n/a"} />
        <Metric label="operation_status" value={overview.operationStatus} />
        <Metric label="target_count" value={overview.targetCount} />
        <Metric label="cycle_count" value={overview.cycleCount} />
        <Metric label="last_updated" value={overview.lastUpdated} />
        <Metric label="last_control_cycle" value={String(overview.lastControlCycle.cycle_index || "n/a")} />
        <Metric label="goal_satisfied" value={overview.goalSatisfied} />
        <Metric label="needs_replan" value={overview.needsReplan} />
      </section>

      <div className="agTimelineGrid">
        <aside className="cycleList" aria-label="Cycle list">
          {cycles.length ? cycles.map((cycle) => (
            <button key={cycle.cycleIndex} className="cycleListItem" onClick={() => cycle.nodes[0] && onSelectNode(cycle.nodes[0])}>
              <span>{cycle.title}</span>
              <small>{cycle.selectedStage} / {cycle.selectedAgent}</small>
              <CycleStatus stopped={cycle.stopped} success={cycle.executionSuccess} />
            </button>
          )) : <div className="emptyState compact">No AG cycles</div>}
        </aside>

        <section className="cycleLanes" aria-label="Cycle execution chains">
          {cycles.length ? cycles.map((cycle) => (
            <article key={cycle.cycleIndex} className="cycleCard">
              <header className="cycleHeader">
                <div>
                  <h2>{cycle.title}</h2>
                  <p>{cycle.selectedStage} / {cycle.selectedAgent}</p>
                </div>
                <dl>
                  <div><dt>execution_success</dt><dd>{cycle.executionSuccess}</dd></div>
                  <div><dt>stopped</dt><dd>{cycle.stopped}</dd></div>
                  <div><dt>stop_reason</dt><dd>{cycle.stopReason}</dd></div>
                </dl>
              </header>
              <div className="executionChain">
                {cycle.nodes.map((node, index) => (
                  <div key={node.id} className="chainStepWrap">
                    <button
                      className={`chainStep ${selectedNode?.id === node.id ? "selected" : ""}`}
                      onClick={() => onSelectNode(node)}
                    >
                      <span className="stepOrder">{inferStepOrder(node)}</span>
                      <strong>{buildDisplayName(node)}</strong>
                      <small>{nodeType(node)}</small>
                      <NodeStatus node={node} />
                    </button>
                    {index < cycle.nodes.length - 1 && <span className="chainArrow">→</span>}
                  </div>
                ))}
              </div>
            </article>
          )) : <div className="emptyState">No attack-process nodes found</div>}

          <section className="statePath" aria-label="Attack state path">
            <h2>攻击状态路径</h2>
            <div className="statePathRow">
              {statePath.length ? statePath.map((node, index) => (
                <div key={node.id} className="statePathItem">
                  <button onClick={() => onSelectNode(node)}>{buildDisplayName(node)}</button>
                  {index < statePath.length - 1 && <span>→</span>}
                </div>
              )) : <p>No state path nodes</p>}
            </div>
          </section>
        </section>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="overviewMetric">
      <span>{label}</span>
      <strong>{String(value)}</strong>
    </div>
  );
}

function CycleStatus({ stopped, success }: { stopped: string; success: string }) {
  if (stopped === "yes") return <AlertTriangle size={16} className="statusBlocked" />;
  if (success === "yes") return <CheckCircle2 size={16} className="statusOk" />;
  return <Clock3 size={16} className="statusPending" />;
}

function NodeStatus({ node }: { node: VisualNode }) {
  const status = String(node.status || readValue(node, "status") || "");
  if (!status) return <Circle size={13} />;
  return <span className={`nodeStatus ${status.toLowerCase()}`}>{status}</span>;
}
