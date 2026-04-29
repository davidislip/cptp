"""Replicate Karamanov (2006) §2.5 (branching) and §3.3 (cut selection) studies
on CPTP instances using the cptp-solve CLI.

For each (instance × config) pair, run cptp-solve with a time limit and
parse the objective, bound, gap, B&B nodes, total cuts, and solve time.
Output a CSV (one row per run) plus a brief Markdown summary of geometric
means of solve time and tree size relative to a baseline configuration.

Usage:
    python experiments/branching_study.py \\
        --instances benchmarks/instances/spprclib \\
        --time_limit 60 \\
        --out experiments/results.csv

    # Replicate just §2 (branching study):
    python experiments/branching_study.py --study branching ...

    # Replicate just §3 (cut selection study):
    python experiments/branching_study.py --study cuts ...

The script tolerates missing data (failed runs) and writes whatever fields
it can extract. Re-runs of the same (instance, config) pair overwrite the
prior row.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Configurations (Karamanov §2 + §3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    name: str
    options: tuple[tuple[str, str], ...]  # (key, value) pairs

    def as_cli_args(self) -> list[str]:
        out: list[str] = []
        for k, v in self.options:
            out += [f"--{k}", v]
        return out


# §2.5: branching study (SIMDI / GENDI / COMBI).
#   SIMDI = baseline = HiGHS variable branching (no hyperplane callback)
#   GENDI = MIG disjunctions only
#   COMBI = mixed (existing pairs + clusters + MIG)
BRANCHING_CONFIGS: list[Config] = [
    Config("SIMDI", (("branch_hyper", "off"),)),
    Config("PAIRS", (("branch_hyper", "pairs"),
                     ("branch_hyper_sb_max_depth", "5"))),
    Config("GENDI", (("branch_hyper", "mig"),
                     ("branch_hyper_mig_k", "10"),
                     ("branch_hyper_sb_max_depth", "5"))),
    Config("COMBI", (("branch_hyper", "all"),
                     ("branch_hyper_mig_k", "10"),
                     ("branch_hyper_sb_max_depth", "5"))),
]

# §3.3: cut selection study (Add-all / DA-dyn(k)).
CUT_SELECTION_CONFIGS: list[Config] = [
    Config("Add-all", (("cut_selector_fraction", "1.0"),)),
    Config("DA-dyn-0.5", (("cut_selector_fraction", "0.5"),)),
    Config("DA-dyn-0.25", (("cut_selector_fraction", "0.25"),)),
    Config("DA-dyn-0.1", (("cut_selector_fraction", "0.1"),)),
]


# ---------------------------------------------------------------------------
# Output parsing (matches benchmarks/run_benchmarks.sh patterns)
# ---------------------------------------------------------------------------

@dataclass
class Result:
    instance: str
    config: str
    objective: float | None = None
    bound: float | None = None
    gap_pct: float | None = None
    time_s: float | None = None
    bb_nodes: int | None = None
    total_cuts: int | None = None
    cut_rounds: int | None = None
    status: str = "ok"  # "ok" | "timeout" | "error"
    stderr_tail: str = ""


_OBJ_RE = re.compile(
    r"Objective:\s*(?P<obj>[-0-9.eE+]+).*Bound:\s*(?P<bnd>[-0-9.eE+]+).*"
    r"Gap:\s*(?P<gap>[-0-9.eE+]+)%?",
    re.DOTALL,
)
_TIME_NODES_RE = re.compile(r"Time:\s*(?P<t>[0-9.eE+]+)s\s+Nodes:\s*(?P<n>\d+)")
_CUTS_RE = re.compile(r"User cuts:\s*(?P<c>\d+)\s*\((?P<r>\d+)\s*rounds?\)")


def _maybe_float(s: str | None) -> float | None:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def parse_output(stdout: str) -> dict:
    """Extract key metrics from cptp-solve stdout."""
    out: dict = {}
    if m := _OBJ_RE.search(stdout):
        out["objective"] = _maybe_float(m.group("obj"))
        out["bound"] = _maybe_float(m.group("bnd"))
        out["gap_pct"] = _maybe_float(m.group("gap"))
    if m := _TIME_NODES_RE.search(stdout):
        out["time_s"] = _maybe_float(m.group("t"))
        try:
            out["bb_nodes"] = int(m.group("n"))
        except ValueError:
            pass
    if m := _CUTS_RE.search(stdout):
        try:
            out["total_cuts"] = int(m.group("c"))
            out["cut_rounds"] = int(m.group("r"))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def find_solver(repo_root: Path) -> Path:
    candidates = [
        repo_root / "build" / "cptp-solve",
        repo_root / "build" / "cptp-solve.exe",
        repo_root / "build" / "Release" / "cptp-solve.exe",
        repo_root / "build" / "Debug" / "cptp-solve.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "cptp-solve not found. Build first:\n"
        "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build -j"
    )


def collect_instances(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files = []
        for ext in (".sppcc", ".vrp", ".txt"):
            files.extend(sorted(target.glob(f"*{ext}")))
        return files
    raise FileNotFoundError(f"Not found: {target}")


def run_one(solver: Path, instance: Path, cfg: Config,
            time_limit: float) -> Result:
    args = [str(solver), str(instance),
            "--time_limit", str(time_limit),
            "--output_flag", "true",
            *cfg.as_cli_args()]

    res = Result(instance=instance.stem, config=cfg.name)
    t0 = time.time()
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=time_limit + 60)
    except subprocess.TimeoutExpired as e:
        res.status = "timeout"
        res.time_s = time.time() - t0
        res.stderr_tail = (e.stderr or "")[-500:]
        return res
    except Exception as e:
        res.status = "error"
        res.stderr_tail = repr(e)[:500]
        return res

    parsed = parse_output(proc.stdout)
    for k, v in parsed.items():
        setattr(res, k, v)
    if proc.returncode != 0:
        res.status = "error"
        res.stderr_tail = (proc.stderr or "")[-500:]
    return res


# ---------------------------------------------------------------------------
# Aggregation: geometric means relative to a baseline
# ---------------------------------------------------------------------------

def geomean(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and v > 0 and math.isfinite(v)]
    if not finite:
        return None
    return math.exp(sum(math.log(v) for v in finite) / len(finite))


def summarize(results: list[Result], baseline: str) -> str:
    """Return a Markdown table of geomeans relative to `baseline` config."""
    by_inst_cfg: dict[tuple[str, str], Result] = {
        (r.instance, r.config): r for r in results
    }
    instances = sorted({r.instance for r in results})
    configs = sorted({r.config for r in results}, key=lambda c: (c != baseline, c))

    def relative(metric: str, cfg: str) -> float | None:
        ratios = []
        for inst in instances:
            r_b = by_inst_cfg.get((inst, baseline))
            r_c = by_inst_cfg.get((inst, cfg))
            if not r_b or not r_c:
                continue
            v_b = getattr(r_b, metric)
            v_c = getattr(r_c, metric)
            if v_b is None or v_c is None or v_b <= 0 or v_c <= 0:
                continue
            ratios.append(v_c / v_b)
        return geomean(ratios)

    lines = [
        f"## Geometric mean ratios relative to baseline `{baseline}`",
        "",
        "| Config | time | nodes | cuts | gap_pct |",
        "|--------|------|-------|------|---------|",
    ]
    for cfg in configs:
        if cfg == baseline:
            continue
        row = [cfg]
        for metric in ("time_s", "bb_nodes", "total_cuts", "gap_pct"):
            r = relative(metric, cfg)
            row.append("—" if r is None else f"{r:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CSV I/O (preserves prior rows for unrelated runs)
# ---------------------------------------------------------------------------

CSV_HEADER = ["instance", "config", "objective", "bound", "gap_pct", "time_s",
              "bb_nodes", "total_cuts", "cut_rounds", "status", "stderr_tail"]


def write_results(out_path: Path, results: list[Result]) -> None:
    keys = {(r.instance, r.config) for r in results}
    existing: list[dict] = []
    if out_path.exists():
        with out_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("instance"), row.get("config")) in keys:
                    continue
                existing.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in existing:
            writer.writerow({k: row.get(k, "") for k in CSV_HEADER})
        for r in results:
            writer.writerow({
                "instance": r.instance,
                "config": r.config,
                "objective": "" if r.objective is None else r.objective,
                "bound": "" if r.bound is None else r.bound,
                "gap_pct": "" if r.gap_pct is None else r.gap_pct,
                "time_s": "" if r.time_s is None else r.time_s,
                "bb_nodes": "" if r.bb_nodes is None else r.bb_nodes,
                "total_cuts": "" if r.total_cuts is None else r.total_cuts,
                "cut_rounds": "" if r.cut_rounds is None else r.cut_rounds,
                "status": r.status,
                "stderr_tail": r.stderr_tail.replace("\n", " ")[:200],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True, type=Path,
                   help="Instance file or directory")
    p.add_argument("--time_limit", type=float, default=60.0,
                   help="Per-run wall-clock limit in seconds (default 60)")
    p.add_argument("--study", choices=("branching", "cuts", "all"),
                   default="all",
                   help="Which configs to run (default: all)")
    p.add_argument("--out", type=Path,
                   default=Path("experiments/results.csv"),
                   help="Output CSV (default experiments/results.csv)")
    p.add_argument("--repo", type=Path,
                   default=Path(__file__).resolve().parents[1],
                   help="Repository root (default: parent of this script's dir)")
    args = p.parse_args()

    solver = find_solver(args.repo)
    instances = collect_instances(args.instances)
    if not instances:
        print("No instances found", file=sys.stderr)
        return 1

    configs: list[Config] = []
    if args.study in ("branching", "all"):
        configs += BRANCHING_CONFIGS
    if args.study in ("cuts", "all"):
        configs += CUT_SELECTION_CONFIGS

    print(f"Solver:    {solver}")
    print(f"Instances: {len(instances)}")
    print(f"Configs:   {[c.name for c in configs]}")
    print(f"Time limit:{args.time_limit}s per run")
    print()

    results: list[Result] = []
    for inst in instances:
        for cfg in configs:
            print(f"  {inst.stem:<30s} {cfg.name:<14s} ", end="", flush=True)
            r = run_one(solver, inst, cfg, args.time_limit)
            results.append(r)
            obj = "—" if r.objective is None else f"{r.objective:.2f}"
            t = "—" if r.time_s is None else f"{r.time_s:.1f}s"
            n = "—" if r.bb_nodes is None else str(r.bb_nodes)
            print(f"obj={obj} time={t} nodes={n} [{r.status}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_results(args.out, results)
    print(f"\nWrote {args.out}")

    if args.study in ("branching", "all"):
        print()
        print(summarize([r for r in results
                         if r.config in {c.name for c in BRANCHING_CONFIGS}],
                        baseline="SIMDI"))
    if args.study in ("cuts", "all"):
        print()
        print(summarize([r for r in results
                         if r.config in {c.name for c in CUT_SELECTION_CONFIGS}],
                        baseline="Add-all"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
