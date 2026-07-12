#!/usr/bin/env python3
"""Generate graphs from experiment summary CSV files."""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate graphs from experiment summary CSV."
    )
    parser.add_argument(
        "--input",
        default="results/experiment_EXP001_summary.csv",
        help="Path to summary CSV (default: results/experiment_EXP001_summary.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/graphs/",
        help="Output directory for PNG graphs (default: results/graphs/)",
    )
    return parser.parse_args()


def read_csv(path):
    """Read CSV and return (header, rows). Raises on errors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with open(path, "r") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {path}")

    header = [h.strip() for h in header]

    required_cols = {
        "eval_step",
        "win_rate",
        "avg_rally_length",
        "avg_hits_per_rally",
        "avg_return_pct",
        "avg_serve_pct",
        "avg_episode_reward",
        "total_proactive",
        "total_passive",
        "avg_distance_at_contact",
    }
    missing = required_cols - set(header)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {', '.join(sorted(missing))}"
        )

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV has header but no data rows: {path}")

    return header, rows


def safe_float(row, col):
    """Convert column value to float, returning None on failure."""
    try:
        val = row[col].strip()
        return float(val) if val else None
    except (ValueError, KeyError):
        return None


def build_plot_data(rows):
    """Extract x (eval_step) and y dicts from rows, skipping bad entries."""
    x_vals = []
    y_data = {col: [] for col in [
        "win_rate",
        "avg_rally_length",
        "avg_hits_per_rally",
        "avg_return_pct",
        "avg_serve_pct",
        "avg_episode_reward",
        "total_proactive",
        "total_passive",
        "avg_distance_at_contact",
    ]}

    for row in rows:
        step = safe_float(row, "eval_step")
        if step is None:
            continue
        x_vals.append(step)

        for col in y_data:
            val = safe_float(row, col)
            if val is not None or col in ("total_proactive", "total_passive"):
                # For aggregate plots we still need None sentinels for alignment
                pass
            y_data[col].append(val)

    # Post-process: for columns where we need paired x/y, remove entries
    # where that specific y is None (but keep x aligned in the process).
    cleaned = {}
    for col in y_data:
        pairs = [(x, y) for x, y in zip(x_vals, y_data[col]) if y is not None]
        if not pairs:
            cleaned[col] = ([], [])
        else:
            cleaned[col] = ([p[0] for p in pairs], [p[1] for p in pairs])

    return cleaned


def save_plot(x, y, xlabel, ylabel, title, output_path):
    """Generate and save a single line plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linestyle="-", markersize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_dual_plot(x1, y1, x2, y2, xlabel, ylabel, title, label1, label2,
                   output_path):
    """Generate and save a plot with two lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x1, y1, marker="o", linestyle="-", markersize=4, label=label1)
    ax.plot(x2, y2, marker="s", linestyle="--", markersize=4, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    # Read data
    try:
        header, rows = read_csv(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Build plot data
    plot_data = build_plot_data(rows)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    xlabel = "Training Steps"
    generated = []

    # --- 1. Win Rate ---
    x, y = plot_data["win_rate"]
    if x:
        path = os.path.join(args.output, "win_rate.png")
        save_plot(x, y, xlabel, "Win Rate", "Win Rate vs Training Steps", path)
        generated.append(("win_rate", len(x)))

    # --- 2. Average Rally Length ---
    x, y = plot_data["avg_rally_length"]
    if x:
        path = os.path.join(args.output, "avg_rally_length.png")
        save_plot(x, y, xlabel, "Avg Rally Length",
                  "Average Rally Length vs Training Steps", path)
        generated.append(("avg_rally_length", len(x)))

    # --- 3. Average Hits Per Rally ---
    x, y = plot_data["avg_hits_per_rally"]
    if x:
        path = os.path.join(args.output, "avg_hits_per_rally.png")
        save_plot(x, y, xlabel, "Avg Hits Per Rally",
                  "Average Hits Per Rally vs Training Steps", path)
        generated.append(("avg_hits_per_rally", len(x)))

    # --- 4. Return Percentage ---
    x, y = plot_data["avg_return_pct"]
    if x:
        path = os.path.join(args.output, "avg_return_pct.png")
        save_plot(x, y, xlabel, "Return %",
                  "Return Percentage vs Training Steps", path)
        generated.append(("avg_return_pct", len(x)))

    # --- 5. Serve Percentage ---
    x, y = plot_data["avg_serve_pct"]
    if x:
        path = os.path.join(args.output, "avg_serve_pct.png")
        save_plot(x, y, xlabel, "Serve %",
                  "Serve Percentage vs Training Steps", path)
        generated.append(("avg_serve_pct", len(x)))

    # --- 6. Proactive vs Passive Interceptions ---
    x1, y1 = plot_data["total_proactive"]
    x2, y2 = plot_data["total_passive"]
    if x1 or x2:
        path = os.path.join(args.output, "proactive_vs_passive.png")
        # Handle case where one has data but the other doesn't
        if not x1:
            x1, y1 = [], []
        if not x2:
            x2, y2 = [], []
        save_dual_plot(
            x1, y1, x2, y2, xlabel, "Count",
            "Proactive vs Passive Interceptions vs Training Steps",
            "Proactive", "Passive", path,
        )
        generated.append(("proactive_vs_passive", max(len(x1), len(x2))))

    # --- 7. Average Distance at Contact ---
    x, y = plot_data["avg_distance_at_contact"]
    if x:
        path = os.path.join(args.output, "avg_distance_at_contact.png")
        save_plot(x, y, xlabel, "Avg Distance at Contact",
                  "Average Distance at Contact vs Training Steps", path)
        generated.append(("avg_distance_at_contact", len(x)))

    # --- 8. Average Episode Reward ---
    x, y = plot_data["avg_episode_reward"]
    if x:
        path = os.path.join(args.output, "avg_episode_reward.png")
        save_plot(x, y, xlabel, "Avg Episode Reward",
                  "Average Episode Reward vs Training Steps", path)
        generated.append(("avg_episode_reward", len(x)))

    # Summary
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Graphs generated ({len(generated)}):")
    for name, pts in generated:
        print(f"  {name}.png  ({pts} data points)")


if __name__ == "__main__":
    main()
