"""
Generate a combined bar/line chart summarizing code generation benchmarks.

Reads the per-model summary CSV (produced by bench.py) and emits a chart that
shows compile success rate alongside judge/model scores for quick comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_chart(summary_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"No rows found in {summary_csv}")
    if "model" not in df.columns:
        raise ValueError("Summary CSV must include a 'model' column.")

    models = df["model"].tolist()
    x_positions = range(len(models))

    fig, ax_left = plt.subplots(figsize=(10, 6))

    bars_plotted = False
    if "compile_success_rate_pct" in df.columns:
        ax_left.bar(
            x_positions,
            df["compile_success_rate_pct"],
            color="#4C72B0",
            alpha=0.7,
            label="Compile success rate (%)",
        )
        ax_left.set_ylabel("Compile success rate (%)")
        bars_plotted = True
    elif "compile_success" in df.columns:
        ax_left.bar(
            x_positions,
            df["compile_success"] * 100,
            color="#4C72B0",
            alpha=0.7,
            label="Compile success rate (%)",
        )
        ax_left.set_ylabel("Compile success rate (%)")
        bars_plotted = True

    ax_right = ax_left.twinx()
    lines_plotted = []

    def plot_line(column: str, label: str, color: str) -> None:
        values = df[column]
        line = ax_right.plot(
            x_positions,
            values,
            marker="o",
            color=color,
            linewidth=2,
            label=label,
        )
        lines_plotted.append(line[0])

    if "spec_compliance_score" in df.columns:
        plot_line("spec_compliance_score", "Spec compliance (avg)", "#55A868")
    if "rougeL_f_vs_spec" in df.columns:
        plot_line("rougeL_f_vs_spec", "ROUGE-L vs spec (avg)", "#C44E52")
    if "token_f1_vs_spec" in df.columns:
        plot_line("token_f1_vs_spec", "Token F1 vs spec (avg)", "#8172B3")

    ax_left.set_xlabel("Model")
    ax_left.set_xticks(list(x_positions))
    ax_left.set_xticklabels(models, rotation=20, ha="right")
    if lines_plotted:
        ax_right.set_ylabel("Score (0-5 scale)" if "spec_compliance_score" in df.columns else "Score")
    elif bars_plotted:
        ax_right.set_ylabel("")

    legend_handles = []
    if bars_plotted:
        legend_handles.append(ax_left.get_children()[0])
    legend_handles.extend(lines_plotted)
    if legend_handles:
        ax_left.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper left")

    ax_left.set_title("Code Generation Benchmark: Model vs Judge Metrics")
    ax_left.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot code generation benchmark summary.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("mba_lm_benchmark") / "results" / "latest_summary.csv",
        help="Path to the summary CSV produced by bench.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mba_lm_benchmark") / "charts" / "code_generation_overview.png",
        help="Output path for the generated chart",
    )
    args = parser.parse_args()
    build_chart(args.summary, args.output)


if __name__ == "__main__":
    main()
