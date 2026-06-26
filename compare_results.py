"""
Compare all experiment results side-by-side.

Scans results/*.json (training metrics) and results/*_eval.json (evaluation metrics)
and prints a ranked comparison table.

Usage:
    python compare_results.py
    python compare_results.py --sort win_rate
    python compare_results.py --min-steps 10000
"""

import os
import json
import glob
import argparse

from config import RESULTS_DIR


def load_all_results(results_dir):
    """Load all training result JSONs, keyed by run tag."""
    runs = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        # Skip eval result files (processed later)
        if fpath.endswith("_eval.json"):
            continue
        with open(fpath) as f:
            data = json.load(f)
        tag = os.path.splitext(os.path.basename(fpath))[0]
        runs[tag] = data
    return runs


def load_eval_results(results_dir):
    """Load all eval result JSONs, keyed by checkpoint name."""
    evals = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*_eval.json"))):
        with open(fpath) as f:
            data = json.load(f)
        ckpt_name = os.path.splitext(os.path.basename(fpath))[0]
        # Strip trailing _eval
        base_name = ckpt_name.replace("_eval", "")
        evals[base_name] = data
    return evals


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--sort", type=str, default="experiment",
                        choices=["experiment", "win_rate", "spin_rate", "steps"],
                        help="Sort key for the table")
    parser.add_argument("--min-steps", type=int, default=0,
                        help="Filter out runs with fewer steps")
    args = parser.parse_args()

    runs = load_all_results(RESULTS_DIR)
    evals = load_eval_results(RESULTS_DIR)

    if not runs:
        print(f"No results found in {RESULTS_DIR}")
        return

    # Build rows
    rows = []
    for tag, train in sorted(runs.items()):
        total_steps = train.get("total_steps", 0)
        if total_steps < args.min_steps:
            continue

        # Training metrics
        config_name = train.get("experiment", tag)
        sparse_name = config_name.split("_")[0][:16]
        steps = train.get("total_steps", "?")
        sps = train.get("steps_per_second", "?")
        ent = train.get("train/entropy_loss", "?")
        ev = train.get("train/explained_variance", "?")
        vl = train.get("train/value_loss", "?")

        # Find matching eval results
        # Try matching by checkpoint tag (which is config_timestamp)
        eval_data = evals.get(tag, None)

        # Also try by base config name
        if eval_data is None:
            for e_tag, e_data in evals.items():
                if tag in e_tag or e_tag in tag:
                    eval_data = e_data
                    break

        if eval_data:
            spin_rate = eval_data.get("spin_rate", {}).get("spin_rate", "?")
            wr_right_noop = eval_data.get("sides", {}).get("right", {}).get("no_op", {}).get("win_rate", "?")
            wr_left_noop = eval_data.get("sides", {}).get("left", {}).get("no_op", {}).get("win_rate", "?")
            wr_right_sq = eval_data.get("sides", {}).get("right", {}).get("square", {}).get("win_rate", "?")
        else:
            spin_rate = "—"
            wr_right_noop = "—"
            wr_left_noop = "—"
            wr_right_sq = "—"

        # Format win rates as percentages
        def fmt_pct(val):
            if val == "?" or val == "—":
                return val
            return f"{val*100:.0f}%"

        rows.append({
            "config": sparse_name,
            "steps": steps,
            "sps": sps if sps == "?" else f"{sps:.0f}",
            "ent": ent if ent == "?" else f"{abs(ent) if isinstance(ent, (int, float)) else ent:.2f}",
            "ev": ev if ev == "?" else f"{ev:.4f}",
            "sr": spin_rate if spin_rate in ("?","—") else f"{spin_rate:.2%}",
            "wr_r_noop": fmt_pct(wr_right_noop),
            "wr_r_sq": fmt_pct(wr_right_sq),
            "wr_l_noop": fmt_pct(wr_left_noop),
            "has_eval": eval_data is not None,
        })

    if not rows:
        print(f"All results filtered out (min_steps={args.min_steps})")
        return

    # Sort
    sort_key = args.sort
    if sort_key == "experiment":
        rows.sort(key=lambda r: r["config"])
    elif sort_key == "win_rate":
        rows.sort(key=lambda r: float(r["wr_r_noop"].rstrip("%"))/100 if r["wr_r_noop"] not in ("?","—") else 0, reverse=True)
    elif sort_key == "spin_rate":
        rows.sort(key=lambda r: float(r["sr"].rstrip("%"))/100 if r["sr"] not in ("?","—") else 1)
    elif sort_key == "steps":
        rows.sort(key=lambda r: r["steps"] if r["steps"] != "?" else 0, reverse=True)

    # Print table
    header = f"{'Config':16s} {'Steps':>6s} {'S/s':>5s} {'Ent':>5s} {'EV':>7s} {'Spin':>6s} | {'W% R→N':>7s} {'W% R→S':>7s} {'W% L→N':>7s}"
    sep = "—" * len(header)
    print(f"\n{'='*len(header)}")
    print("  Experiment Comparison")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in rows:
        print(f"{r['config']:16s} {r['steps']:>6} {r['sps']:>5} {r['ent']:>5} {r['ev']:>7} {r['sr']:>6} | "
              f"{r['wr_r_noop']:>7} {r['wr_r_sq']:>7} {r['wr_l_noop']:>7}")
    print(sep)
    print(f"\n{'='*len(header)}")
    print("  W% R→N = Win rate, Right side vs no_op")
    print("  W% R→S = Win rate, Right side vs square")
    print("  W% L→N = Win rate, Left side vs no_op")
    print("  Spin   = Fraction of steps with non-neutral rotation")
    print(f"{'='*len(header)}")
    print(f"\n{len(rows)} experiments. {sum(1 for r in rows if r['has_eval'])} with eval data.")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
