"""High-level convenience wrapper around the C++ CPTP solver."""

import numpy as np
from cptp._cptp import Model as _Model, SolveResult


def solve(
    num_nodes: int,
    edges: np.ndarray,
    edge_costs: np.ndarray,
    profits: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    depot: int = 0,
    source: int | None = None,
    target: int | None = None,
    time_limit: float = 600.0,
    num_threads: int | None = None,
    verbose: bool = False,
    branch_hyper: str | None = None,
    branch_hyper_mig_k: int | None = None,
    branch_hyper_sb_max_depth: int | None = None,
    branch_hyper_sb_iter_limit: int | None = None,
    branch_hyper_sb_min_reliable: int | None = None,
    branch_hyper_sb_max_candidates: int | None = None,
    cut_selector_fraction: float | None = None,
) -> SolveResult:
    """Solve a CPTP instance.

    Args:
        num_nodes: Number of nodes in the graph.
        edges: (m, 2) array of (tail, head) pairs.
        edge_costs: (m,) array of edge costs (can be negative).
        profits: (n,) array of node profits.
        demands: (n,) array of node demands.
        capacity: Vehicle capacity.
        depot: Depot node index (default 0). Sets source = target = depot (tour).
        source: Source node for s-t path. Overrides depot if set.
        target: Target node for s-t path. Overrides depot if set.
        time_limit: Time limit in seconds.
        num_threads: Number of threads.
        verbose: Print solver output.
        branch_hyper: Hyperplane branching mode: "off", "pairs", "clusters",
            "demand", "cardinality", "mig", or "all".
        branch_hyper_mig_k: Top-k MIG disjunctions per node (default 10).
        branch_hyper_sb_max_depth: Strong-branching depth limit (default 0).
        branch_hyper_sb_iter_limit: Simplex iterations per SB trial (default 100).
        branch_hyper_sb_min_reliable: Pseudocost samples before trusted (default 4).
        branch_hyper_sb_max_candidates: Top-k for SB evaluation (default 3).

    Returns:
        SolveResult with tour, objective, gap, etc.
    """
    model = _Model()
    model.set_graph(
        num_nodes,
        np.ascontiguousarray(edges, dtype=np.int32),
        np.ascontiguousarray(edge_costs, dtype=np.float64),
    )

    if source is not None or target is not None:
        model.set_source(source if source is not None else depot)
        model.set_target(target if target is not None else depot)
    else:
        model.set_depot(depot)

    model.set_profits(np.ascontiguousarray(profits, dtype=np.float64))
    model.add_capacity_resource(np.ascontiguousarray(demands, dtype=np.float64), capacity)

    options: list[tuple[str, str]] = [
        ("time_limit", str(time_limit)),
        ("output_flag", "true" if verbose else "false"),
    ]
    if num_threads is not None:
        options.append(("threads", str(num_threads)))
    if branch_hyper is not None:
        options.append(("branch_hyper", branch_hyper))
    if branch_hyper_mig_k is not None:
        options.append(("branch_hyper_mig_k", str(branch_hyper_mig_k)))
    if branch_hyper_sb_max_depth is not None:
        options.append(("branch_hyper_sb_max_depth", str(branch_hyper_sb_max_depth)))
    if branch_hyper_sb_iter_limit is not None:
        options.append(("branch_hyper_sb_iter_limit", str(branch_hyper_sb_iter_limit)))
    if branch_hyper_sb_min_reliable is not None:
        options.append(("branch_hyper_sb_min_reliable", str(branch_hyper_sb_min_reliable)))
    if branch_hyper_sb_max_candidates is not None:
        options.append(("branch_hyper_sb_max_candidates", str(branch_hyper_sb_max_candidates)))
    if cut_selector_fraction is not None:
        options.append(("cut_selector_fraction", str(cut_selector_fraction)))
    return model.solve(options)
