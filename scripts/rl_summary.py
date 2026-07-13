"""Aggregate the RL sweep results into a readable summary + per-seed CSV.

Reads every <run>/eval_dr/rl_evaluation.json and <run>/eval_nom/rl_evaluation.json
under the sweep root, groups runs by configuration (tag without the seed),
and reports, per configuration: number of seeds, mean / std / max of the DR
success rate, how many seeds reach 100 % nominal suppression, and the mean DR
cost. Run directories are named ``tp<TP>_ts<M>M_lr<SCHED>_s<SEED>``.

Usage:
    poetry run python scripts/rl_summary.py results/rl/sweep
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

TAG_RE = re.compile(r"^(tp\d+_ts\d+M_lr[a-z]+(?:_ent[0-9.]+)?)_s(\d+)$")


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "results/rl/sweep")
    rows: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = TAG_RE.match(d.name)
        if not m:
            continue
        dr = _load(d / "eval_dr" / "rl_evaluation.json")
        nom = _load(d / "eval_nom" / "rl_evaluation.json")
        if dr is None:
            continue
        rows.append(
            {
                "config": m.group(1),
                "seed": int(m.group(2)),
                "dr_succ": float(dr.get("success_rate", float("nan"))),
                "dr_cost": float(dr.get("mean_cost", float("nan"))),
                "nom_succ": float(nom.get("success_rate", float("nan")))
                if nom
                else float("nan"),
                "nom_cost": float(nom.get("mean_cost", float("nan")))
                if nom
                else float("nan"),
            }
        )

    if not rows:
        print(f"No evaluated results found under {root}")
        return 1

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["config"]].append(r)

    print(f"\nRL SWEEP — summary ({len(rows)} runs, {len(groups)} configs)\n")
    hdr = (
        f"{'config':28} {'n':>2} {'DRsucc_mean':>11} {'DRsucc_std':>10} "
        f"{'DRsucc_max':>10} {'nom100%':>8} {'DRcost_mean':>12} {'NOMcost_best':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    # sort by mean DR success descending
    def _key(item: tuple[str, list[dict]]) -> float:
        return -mean([r["dr_succ"] for r in item[1]])

    for cfg, rs in sorted(groups.items(), key=_key):
        ds = [r["dr_succ"] for r in rs]
        n100 = sum(1 for r in rs if r["nom_succ"] >= 0.999)
        drc = mean([r["dr_cost"] for r in rs])
        # best NOM cost among seeds that fully suppress nominally
        succ_costs = [r["nom_cost"] for r in rs if r["nom_succ"] >= 0.999]
        best_nom = min(succ_costs) if succ_costs else float("nan")
        print(
            f"{cfg:28} {len(rs):>2} {mean(ds):>11.3f} "
            f"{(pstdev(ds) if len(ds) > 1 else 0.0):>10.3f} {max(ds):>10.3f} "
            f"{n100:>8} {drc:>12.3e} {best_nom:>12.3e}"
        )

    # best individual policy overall (by DR success, tie-break lower DR cost)
    best = max(rows, key=lambda r: (r["dr_succ"], -r["dr_cost"]))
    print(
        f"\nBest individual policy: {best['config']}_s{best['seed']}  "
        f"DRsucc={best['dr_succ']:.3f}  DRcost={best['dr_cost']:.3e}  "
        f"NOMsucc={best['nom_succ']:.3f}  NOMcost={best['nom_cost']:.3e}"
    )

    csv = root / "per_seed.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("config,seed,dr_success,dr_cost,nom_success,nom_cost\n")
        for r in sorted(rows, key=lambda r: (r["config"], r["seed"])):
            f.write(
                f"{r['config']},{r['seed']},{r['dr_succ']},{r['dr_cost']},"
                f"{r['nom_succ']},{r['nom_cost']}\n"
            )
    print(f"\nPer-seed detail: {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
