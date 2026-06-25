"""Generic predicate engine for evaluating success contract conditions.

All predicates operate on KG nodes/edges, AG nodes/edges, and RuntimeState.
Zone references are resolved via OperationProfile.zone_bindings, never by
hardcoded network names or CIDRs.

No environment-specific values (IPs, hostnames, flag paths, exploit names)
may appear in this module.
"""

from __future__ import annotations

import ipaddress
from typing import Any, Callable

from src.core.evaluation.models import ConditionResult, OperationProfile


def _parse_ip(value: Any) -> Any:
    """Parse ``value`` as an IP address or network, tolerating a ``:port`` suffix.

    Returns an ``ip_address``/``ip_network`` object, or ``None`` if the value is
    not locatable as an IP/CIDR (e.g. a free-text evidence summary).
    """

    text = str(value or "").strip()
    if not text:
        return None
    candidates = [text]
    if ":" in text and "/" not in text:
        candidates.append(text.rsplit(":", 1)[0])  # strip a trailing :port
    for cand in candidates:
        try:
            return ipaddress.ip_network(cand, strict=False) if "/" in cand else ipaddress.ip_address(cand)
        except ValueError:
            continue
    return None


def _addr_in_cidrs(value: Any, cidrs: list[str]) -> bool:
    """Return True if ``value`` is an IP/CIDR contained in any of ``cidrs``.

    Generic CIDR containment used to resolve a node/route to a logical zone via
    the profile's zone CIDRs. The CIDRs come from the profile, never hardcoded.
    """

    if not cidrs:
        return False
    candidate = _parse_ip(value)
    if candidate is None:
        return False
    for raw in cidrs:
        try:
            network = ipaddress.ip_network(str(raw), strict=False)
        except ValueError:
            continue
        if isinstance(candidate, ipaddress._BaseNetwork):
            if candidate.subnet_of(network):
                return True
        elif candidate in network:
            return True
    return False


def _node_addresses(node: dict[str, Any]) -> list[str]:
    """Collect address-like strings from a node for CIDR-based zone resolution."""

    props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    values = [
        node.get("address"),
        node.get("host"),
        node.get("ip"),
        node.get("hostname"),
        node.get("label"),
        props.get("address"),
        props.get("host"),
        props.get("ip"),
    ]
    return [str(v) for v in values if v]


# ---------------------------------------------------------------------------
# Type alias for predicate implementations
# ---------------------------------------------------------------------------

PredicateFn = Callable[["PredicateContext", dict[str, Any]], ConditionResult]

_REGISTRY: dict[str, PredicateFn] = {}


def _register(name: str) -> Callable[[PredicateFn], PredicateFn]:
    def decorator(fn: PredicateFn) -> PredicateFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Context object passed into every predicate
# ---------------------------------------------------------------------------


class PredicateContext:
    """Runtime context injected into every predicate call.

    Holds KG/AG/Runtime snapshots and the OperationProfile for zone resolution.
    No hardcoded environment values here.
    """

    def __init__(
        self,
        *,
        profile: OperationProfile,
        kg_nodes: list[dict[str, Any]] | None = None,
        kg_edges: list[dict[str, Any]] | None = None,
        ag_nodes: list[dict[str, Any]] | None = None,
        ag_edges: list[dict[str, Any]] | None = None,
        runtime_state: dict[str, Any] | None = None,
        oracle_results: dict[str, bool] | None = None,
    ) -> None:
        self.profile = profile
        self.kg_nodes: list[dict[str, Any]] = kg_nodes or []
        self.kg_edges: list[dict[str, Any]] = kg_edges or []
        self.ag_nodes: list[dict[str, Any]] = ag_nodes or []
        self.ag_edges: list[dict[str, Any]] = ag_edges or []
        self.runtime_state: dict[str, Any] = runtime_state or {}
        # Pre-evaluated oracle results keyed by goal_id
        self.oracle_results: dict[str, bool] = oracle_results or {}

    # ------------------------------------------------------------------
    # Graph query helpers
    # ------------------------------------------------------------------

    def nodes_from(self, graph: str) -> list[dict[str, Any]]:
        if graph == "kg":
            return self.kg_nodes
        if graph == "ag":
            return self.ag_nodes
        if graph == "runtime":
            return self._runtime_nodes()
        return []

    def edges_from(self, graph: str) -> list[dict[str, Any]]:
        if graph == "kg":
            return self.kg_edges
        if graph == "ag":
            return self.ag_edges
        return []

    def _runtime_nodes(self) -> list[dict[str, Any]]:
        """Flatten pivot_routes, sessions, credentials from runtime as pseudo-nodes."""
        nodes: list[dict[str, Any]] = []
        for route_id, route in (self.runtime_state.get("pivot_routes") or {}).items():
            nodes.append({"type": "PivotRoute", "route_id": route_id, **route})
        for sess_id, sess in (self.runtime_state.get("sessions") or {}).items():
            nodes.append({"type": "Session", "session_id": sess_id, **sess})
        for cred_id, cred in (self.runtime_state.get("credentials") or {}).items():
            nodes.append({"type": "Credential", "credential_id": cred_id, **cred})
        return nodes

    def resolve_zone_cidrs(self, zone_ref: str) -> list[str]:
        """Return CIDRs for a logical zone reference via profile.zone_bindings."""
        zone = self.profile.resolve_zone(zone_ref)
        return zone.cidrs if zone else []

    def pivot_routes_for_zone(self, zone_ref: str) -> list[dict[str, Any]]:
        """Return active pivot routes that target the given zone_ref."""
        zone = self.profile.resolve_zone(zone_ref)
        if zone is None:
            return []
        target_cidrs = list(zone.cidrs)
        target_name = zone.name.lower()
        routes = []
        for route in (self.runtime_state.get("pivot_routes") or {}).values():
            if not isinstance(route, dict):
                continue
            if route.get("status") not in ("active", "verified", "ACTIVE"):
                continue
            dest_zone = str(route.get("destination_zone") or "").lower()
            dest_cidr = str(route.get("destination_cidr") or "")
            dest_host = route.get("destination_host")
            if (
                dest_zone in (target_name, zone_ref.lower())
                or dest_cidr in set(target_cidrs)
                or _addr_in_cidrs(dest_cidr, target_cidrs)
                or _addr_in_cidrs(dest_host, target_cidrs)
            ):
                routes.append(route)
        # Also check KG PivotRoute nodes
        for node in self.kg_nodes:
            if node.get("type") != "PivotRoute":
                continue
            if node.get("status") not in ("active", "verified", None):
                continue
            if node.get("to_zone_ref") == zone_ref:
                routes.append(node)
        return routes


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------


def _match_type(node: dict[str, Any], type_spec: Any) -> bool:
    node_type = str(node.get("type") or "")
    if isinstance(type_spec, dict):
        allowed = {str(t) for t in (type_spec.get("in") or [])}
        return node_type in allowed
    if isinstance(type_spec, list):
        return node_type in {str(t) for t in type_spec}
    return node_type == str(type_spec)


def _match_filters(node: dict[str, Any], filters: dict[str, Any], ctx: PredicateContext) -> bool:
    for key, spec in filters.items():
        if key == "zone_ref":
            # Resolve zone_ref → CIDRs/name, then check node properties
            if not _match_zone_ref(node, spec, ctx):
                return False
            continue
        val = node.get(key)
        if val is None:
            val = (node.get("properties") or {}).get(key)
        if isinstance(spec, dict):
            in_vals = spec.get("in")
            gte_val = spec.get("gte")
            lte_val = spec.get("lte")
            if in_vals is not None and str(val) not in {str(v) for v in in_vals}:
                return False
            if gte_val is not None and (val is None or val < gte_val):
                return False
            if lte_val is not None and (val is None or val > lte_val):
                return False
        else:
            if str(val) != str(spec):
                return False
    return True


def _match_zone_ref(node: dict[str, Any], zone_ref: str, ctx: PredicateContext) -> bool:
    """Check whether a node belongs to the specified zone_ref.

    Checks node.zone_ref property first, then falls back to CIDR matching
    against node address/hostname if the zone has CIDRs defined.
    """
    node_zone = node.get("zone_ref") or (node.get("properties") or {}).get("zone_ref")
    if node_zone is not None:
        if str(node_zone) == zone_ref:
            return True
        # Node carries a zone tag that names the zone rather than its ref.
        zone = ctx.profile.resolve_zone(zone_ref)
        if zone is not None and str(node_zone).lower() == zone.name.lower():
            return True
        # A mismatching explicit tag is only authoritative when no CIDR can
        # confirm membership; fall through to CIDR resolution below.
    # CIDR fallback: resolve the node's address against the zone's profile CIDRs.
    # Only nodes that actually carry a locatable IP address are placed (or excluded)
    # by CIDR. Address-less nodes (free-text Evidence/Observation/Finding) cannot be
    # disproven, so they stay permissive — matching the engine's pre-CIDR behavior.
    cidrs = ctx.resolve_zone_cidrs(zone_ref)
    located = [addr for addr in _node_addresses(node) if _parse_ip(addr) is not None]
    if cidrs and located:
        return any(_addr_in_cidrs(addr, cidrs) for addr in located)
    # No locatable address (or no CIDRs): permissive unless an explicit, mismatching
    # zone tag was present (handled above → node_zone set means exclude).
    return node_zone is None


def _count_evidence(node: dict[str, Any]) -> int:
    ev = node.get("evidence_ids") or node.get("evidence_refs") or []
    return len(ev)


# ---------------------------------------------------------------------------
# Predicate implementations
# ---------------------------------------------------------------------------


@_register("exists_node")
def _exists_node(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    graph = str(args.get("graph") or "kg")
    type_spec = args.get("type")
    filters = dict(args.get("filters") or {})
    nodes = ctx.nodes_from(graph)
    matched = []
    for node in nodes:
        if type_spec and not _match_type(node, type_spec):
            continue
        if filters and not _match_filters(node, filters, ctx):
            continue
        matched.append(str(node.get("id") or node.get("route_id") or node.get("session_id") or ""))
    satisfied = len(matched) > 0
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="exists_node",
        matched_node_ids=matched,
        redacted_summary=f"Found {len(matched)} node(s)" if satisfied else "No matching node found",
    )


@_register("count_nodes_at_least")
def _count_nodes_at_least(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    graph = str(args.get("graph") or "kg")
    type_spec = args.get("type")
    filters = dict(args.get("filters") or {})
    min_count = int(args.get("min_count") or 1)
    nodes = ctx.nodes_from(graph)
    matched = [
        str(n.get("id") or "")
        for n in nodes
        if (not type_spec or _match_type(n, type_spec))
        and (not filters or _match_filters(n, filters, ctx))
    ]
    satisfied = len(matched) >= min_count
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="count_nodes_at_least",
        matched_node_ids=matched,
        redacted_summary=f"{len(matched)}/{min_count} node(s) required",
    )


@_register("exists_edge")
def _exists_edge(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    graph = str(args.get("graph") or "kg")
    edge_type = args.get("type")
    source_filter = dict(args.get("source_filter") or {})
    target_filter = dict(args.get("target_filter") or {})
    edges = ctx.edges_from(graph)
    nodes_by_id = {str(n.get("id") or ""): n for n in ctx.nodes_from(graph)}
    matched_ids = []
    for edge in edges:
        if edge_type and str(edge.get("type") or "") != str(edge_type):
            continue
        src = nodes_by_id.get(str(edge.get("source") or ""), {})
        tgt = nodes_by_id.get(str(edge.get("target") or ""), {})
        if source_filter and not _match_filters(src, source_filter, ctx):
            continue
        if target_filter and not _match_filters(tgt, target_filter, ctx):
            continue
        matched_ids.append(str(edge.get("id") or ""))
    satisfied = len(matched_ids) > 0
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="exists_edge",
        matched_edge_ids=matched_ids,
        redacted_summary=f"Found {len(matched_ids)} edge(s)" if satisfied else "No matching edge",
    )


@_register("path_exists")
def _path_exists(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    graph = str(args.get("graph") or "kg")
    start_filter = dict(args.get("start_filter") or {})
    end_filter = dict(args.get("end_filter") or {})
    allowed_edge_types = set(args.get("edge_types") or [])
    nodes = ctx.nodes_from(graph)
    edges = ctx.edges_from(graph)
    start_ids = {str(n.get("id") or "") for n in nodes if _match_filters(n, start_filter, ctx)}
    end_ids = {str(n.get("id") or "") for n in nodes if _match_filters(n, end_filter, ctx)}
    # Build adjacency
    adj: dict[str, set[str]] = {}
    for edge in edges:
        if allowed_edge_types and str(edge.get("type") or "") not in allowed_edge_types:
            continue
        src = str(edge.get("source") or "")
        tgt = str(edge.get("target") or "")
        adj.setdefault(src, set()).add(tgt)
    # BFS from start_ids
    visited = set(start_ids)
    frontier = set(start_ids)
    while frontier:
        next_frontier: set[str] = set()
        for node_id in frontier:
            for neighbor in adj.get(node_id, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    found = end_ids & visited
    satisfied = len(found) > 0
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="path_exists",
        matched_node_ids=list(found),
        redacted_summary=f"Path found to {len(found)} end node(s)" if satisfied else "No path found",
    )


@_register("node_has_evidence")
def _node_has_evidence(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    graph = str(args.get("graph") or "kg")
    node_filter = dict(args.get("node_filter") or {})
    min_evidence_count = int(args.get("min_evidence_count") or 1)
    nodes = ctx.nodes_from(graph)
    matched = []
    for node in nodes:
        if node_filter and not _match_filters(node, node_filter, ctx):
            continue
        if _count_evidence(node) >= min_evidence_count:
            matched.append(str(node.get("id") or ""))
    satisfied = len(matched) > 0
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="node_has_evidence",
        matched_node_ids=matched,
        redacted_summary=f"{len(matched)} node(s) with sufficient evidence" if satisfied else "No node with enough evidence",
    )


@_register("service_discovered_via_route")
def _service_discovered_via_route(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    """Check that a Service exists in a zone reachable only via pivot route.

    Generic: resolves zone_ref → zone definition from profile, never hardcodes
    network names.
    """
    zone_ref = str(args.get("zone_ref") or "")
    route_type = str(args.get("route_type") or "pivot_route")

    if not zone_ref:
        return ConditionResult(
            condition="",
            satisfied=False,
            predicate="service_discovered_via_route",
            error="zone_ref argument is required",
        )

    # 1. Must have at least one active route to this zone
    routes = ctx.pivot_routes_for_zone(zone_ref)
    if not routes:
        return ConditionResult(
            condition="",
            satisfied=False,
            predicate="service_discovered_via_route",
            redacted_summary=f"No active {route_type} route to zone '{zone_ref}'",
        )

    # 2. Must have at least one Service node associated with this zone
    zone = ctx.profile.resolve_zone(zone_ref)
    zone_name = zone.name.lower() if zone else zone_ref.lower()

    zone_cidrs = ctx.resolve_zone_cidrs(zone_ref)
    matched_services = []
    for node in ctx.kg_nodes:
        if node.get("type") not in ("Service", "InternalService"):
            continue
        node_zone = str(node.get("zone_ref") or (node.get("properties") or {}).get("zone_ref") or "").lower()
        if node_zone in (zone_ref.lower(), zone_name):
            matched_services.append(str(node.get("id") or ""))
            continue
        # CIDR fallback: a service at an address inside the zone's profile CIDRs
        # belongs to the zone even when it carries no explicit zone tag.
        if not node_zone and zone_cidrs and any(_addr_in_cidrs(addr, zone_cidrs) for addr in _node_addresses(node)):
            matched_services.append(str(node.get("id") or ""))

    satisfied = len(matched_services) > 0
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="service_discovered_via_route",
        matched_node_ids=matched_services,
        redacted_summary=(
            f"Found {len(matched_services)} service(s) in zone '{zone_ref}' via {route_type}"
            if satisfied
            else f"No service discovered in zone '{zone_ref}' via pivot route"
        ),
    )


@_register("route_authorized")
def _route_authorized(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    from_zone_ref = str(args.get("from_zone_ref") or "")
    to_zone_ref = str(args.get("to_zone_ref") or "")
    routes = ctx.pivot_routes_for_zone(to_zone_ref)
    zone = ctx.profile.resolve_zone(to_zone_ref)
    requires = zone.requires if zone else []
    route_satisfied = len(routes) > 0
    # Check that required capabilities are also present
    creds_or_sessions = ctx.runtime_state.get("sessions") or {}
    capability_ok = True
    if "capability" in requires:
        active_sessions = [s for s in creds_or_sessions.values() if isinstance(s, dict) and s.get("status") == "active"]
        capability_ok = len(active_sessions) > 0
    satisfied = route_satisfied and capability_ok
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="route_authorized",
        redacted_summary=f"Route from {from_zone_ref} to {to_zone_ref}: {'authorized' if satisfied else 'not authorized'}",
    )


@_register("oracle_proof_valid")
def _oracle_proof_valid(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    """Check whether a GoalOracle proof has been pre-validated and recorded."""
    goal_id = str(args.get("goal_id") or "")
    if not goal_id:
        return ConditionResult(
            condition="",
            satisfied=False,
            predicate="oracle_proof_valid",
            error="goal_id argument is required",
        )
    # Check oracle_results cache (populated by GoalOracle before predicate evaluation)
    passed = ctx.oracle_results.get(goal_id, False)
    # Also check KG GoalProof nodes
    if not passed:
        for node in ctx.kg_nodes:
            if node.get("type") == "GoalProof" and node.get("goal_id") == goal_id:
                passed = True
                break
    return ConditionResult(
        condition="",
        satisfied=passed,
        predicate="oracle_proof_valid",
        redacted_summary="Goal proof validated" if passed else "Goal proof not yet valid",
    )


@_register("chain_satisfied")
def _chain_satisfied(ctx: PredicateContext, args: dict[str, Any]) -> ConditionResult:
    """Verify that a list of conditions are satisfied in documented chain order."""
    # Delegate to individual condition results stored in ctx
    # This predicate is evaluated last, after all other conditions
    chain_results = list(args.get("chain_results") or [])
    satisfied = all(chain_results) if chain_results else False
    return ConditionResult(
        condition="",
        satisfied=satisfied,
        predicate="chain_satisfied",
        redacted_summary=f"Chain: {sum(chain_results)}/{len(chain_results)} conditions satisfied",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PredicateEngine:
    """Evaluate named predicates against KG/AG/Runtime context.

    Usage:
        engine = PredicateEngine()
        result = engine.evaluate("exists_node", {"graph": "kg", "type": "Host"}, ctx)
    """

    def evaluate(
        self,
        predicate_name: str,
        args: dict[str, Any],
        ctx: PredicateContext,
        condition_name: str = "",
    ) -> ConditionResult:
        fn = _REGISTRY.get(predicate_name)
        if fn is None:
            return ConditionResult(
                condition=condition_name,
                satisfied=False,
                predicate=predicate_name,
                error=f"Unknown predicate: '{predicate_name}'. This is a configuration error.",
            )
        result = fn(ctx, args)
        result.condition = condition_name
        return result

    @property
    def known_predicates(self) -> list[str]:
        return list(_REGISTRY.keys())
