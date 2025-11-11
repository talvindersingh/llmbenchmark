#!/usr/bin/env python3
"""
Generate professional charts from code-generation benchmarking CSVs.

Usage:
    python plot_benchmarks.py --summary code_gen_node_java_py_summary.csv --detail code_gen_node_java_py.csv --outdir plots

Design:
- Uses matplotlib (no seaborn).
- Each chart is a separate figure (no subplots).
- Cycles through curated color palettes for readability.
- Works with "summary" (one row per model) and "detail" (one row per task/model) files.
- Groups by `model` if no language column is available.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from itertools import cycle

def safe_title(s: str):
    return "".join([ch if ch.isalnum() or ch in "._- " else "_" for ch in s])

def annotate_bars(values):
    for i, v in enumerate(values):
        try:
            label = f"{v:.3g}"
        except Exception:
            label = str(v)
        plt.text(i, v, label, ha='center', va='bottom', fontsize=9)

def color_sequence(n: int, cmap_name: str = "tab20"):
    if n <= 0:
        return []
    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors") and len(cmap.colors) >= n:
        return cmap.colors[:n]
    return [cmap(x) for x in np.linspace(0, 1, n)]

PALETTE_NAMES = [
    "tab10",
    "Set2",
    "Dark2",
    "Set3",
    "Accent",
    "Pastel1",
    "Pastel2",
    "tab20",
    "tab20b",
    "tab20c",
]
PALETTE_CYCLE = cycle(PALETTE_NAMES)

def next_color_sequence(n: int, cmap_name: Optional[str] = None):
    try:
        palette = cmap_name if cmap_name is not None else next(PALETTE_CYCLE)
    except Exception:
        palette = "tab20"
    return color_sequence(n, palette)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=False, default="code_gen_node_java_py_summary.csv")
    ap.add_argument("--detail", type=str, required=False, default="code_gen_node_java_py.csv")
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    detail_path = Path(args.detail)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfs = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    df = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()

    # Determine grouping column (language/model/etc.)
    group_col = None
    for cand in ["language", "lang", "sdk", "runtime", "env", "model"]:
        if cand in df.columns:
            group_col = cand
            break
    if group_col is None:
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                group_col = c
                break

    generated = []

    # ------------------ SUMMARY PLOTS ------------------
    if not dfs.empty and "model" in dfs.columns:
        num_cols = dfs.select_dtypes(include=[np.number]).columns.tolist()
        for metric in num_cols:
            series = dfs[["model", metric]].dropna()
            if series.empty:
                continue
            # Drop negative sentinel values (e.g., -1.0 meaning "not measured")
            if (series[metric] < 0).sum() > 0:
                series = series[series[metric] >= 0]
            if series.empty:
                continue
            series = series.sort_values(metric, ascending=False)
            plt.figure(figsize=(10, 5.8))
            colors = next_color_sequence(len(series))
            plt.bar(series["model"].astype(str), series[metric], color=colors, edgecolor="black", linewidth=0.6)
            plt.title(f"{metric} by model (summary)")
            plt.xlabel("model")
            plt.ylabel(metric)
            plt.grid(True, axis='y', linestyle='--', alpha=0.4)
            annotate_bars(series[metric].values)
            plt.tight_layout()
            out = outdir / f"summary_{safe_title(metric)}_by_model.png"
            plt.savefig(out, dpi=220); plt.close()
            generated.append(str(out))

    # ------------------ DETAIL PLOTS ------------------
    if not df.empty and group_col:
        handled_metrics = set()
        numeric_series = {}
        for col in df.columns:
            if col == group_col or col == "compile_success":
                continue
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors="coerce")
            if isinstance(series, pd.Series) and series.notna().sum() > 0:
                numeric_series[col] = series

        # 1) Compile success rate
        if "compile_success" in df.columns:
            s = df["compile_success"]
            try:
                s = pd.to_numeric(s, errors="coerce")
            except Exception:
                s = s.astype(str).str.lower().map({"true":1,"false":0,"yes":1,"no":0,"pass":1,"fail":0})
            tmp = pd.DataFrame({group_col: df[group_col], "success": s})
            grp = tmp.groupby(group_col)["success"].mean().reset_index().dropna().sort_values("success", ascending=False)
            if not grp.empty:
                plt.figure(figsize=(10, 5.8))
                colors = next_color_sequence(len(grp))
                plt.bar(grp[group_col].astype(str), grp["success"] * 100.0, color=colors, edgecolor="black", linewidth=0.6)
                plt.title(f"Compile Success Rate by {group_col} (%)")
                plt.xlabel(group_col); plt.ylabel("Success Rate (%)")
                plt.grid(True, axis='y', linestyle='--', alpha=0.4)
                annotate_bars((grp["success"] * 100.0).values)
                plt.tight_layout()
                out = outdir / f"detail_compile_success_rate_by_{safe_title(group_col)}.png"
                plt.savefig(out, dpi=220); plt.close()
                generated.append(str(out))
                handled_metrics.add("compile_success")

        # 2) Runtime distributions (boxplot): wall_time_ms -> total_duration_ms -> eval_duration_ms
        runtime_candidates = [c for c in ["wall_time_ms", "total_duration_ms", "eval_duration_ms"] if c in df.columns]
        if runtime_candidates:
            runtime_col = runtime_candidates[0]
            r = pd.to_numeric(df[runtime_col], errors="coerce")
            tmp = pd.DataFrame({group_col: df[group_col], "runtime": r}).dropna()
            if not tmp.empty and tmp["runtime"].notna().sum() > 0:
                order = tmp.groupby(group_col)["runtime"].median().sort_values().index.tolist()
                data = [tmp[tmp[group_col] == l]["runtime"].values for l in order]
                plt.figure(figsize=(10, 5.8))
                colors = next_color_sequence(len(data))
                bp = plt.boxplot(
                    data,
                    labels=[str(l) for l in order],
                    showmeans=True,
                    patch_artist=True,
                )
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                for median in bp["medians"]:
                    median.set_color("#333333")
                plt.title(f"{runtime_col} distribution by {group_col}")
                plt.xlabel(group_col); plt.ylabel(runtime_col)
                plt.grid(True, axis='y', linestyle='--', alpha=0.4)
                plt.tight_layout()
                out = outdir / f"detail_{safe_title(runtime_col)}_distribution_by_{safe_title(group_col)}.png"
                plt.savefig(out, dpi=220); plt.close()
                generated.append(str(out))
                handled_metrics.add(runtime_col)

        # 3) Generic numeric metrics (mean per group)
        for metric, series in numeric_series.items():
            if metric in handled_metrics:
                continue
            data = pd.DataFrame({group_col: df[group_col], metric: series})
            grp = data.groupby(group_col)[metric].mean().reset_index().dropna()
            if grp.empty:
                continue
            grp = grp.sort_values(metric, ascending=False)
            plt.figure(figsize=(10, 5.8))
            colors = next_color_sequence(len(grp))
            plt.bar(grp[group_col].astype(str), grp[metric], color=colors, edgecolor="black", linewidth=0.6)
            plt.title(f"{metric} by {group_col} (mean)")
            plt.xlabel(group_col); plt.ylabel(metric)
            plt.grid(True, axis='y', linestyle='--', alpha=0.4)
            annotate_bars(grp[metric].values)
            plt.tight_layout()
            out = outdir / f"detail_{safe_title(metric)}_by_{safe_title(group_col)}.png"
            plt.savefig(out, dpi=220); plt.close()
            generated.append(str(out))

        # 4) Volume per group
        counts = df[group_col].astype(str).value_counts().reset_index()
        counts.columns = [group_col, "count"]
        plt.figure(figsize=(10, 5.8))
        colors = next_color_sequence(len(counts))
        plt.bar(counts[group_col].astype(str), counts["count"], color=colors, edgecolor="black", linewidth=0.6)
        plt.title(f"Number of Evaluations by {group_col}")
        plt.xlabel(group_col); plt.ylabel("count")
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        for i, v in enumerate(counts["count"]):
            plt.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        out = outdir / f"detail_counts_by_{safe_title(group_col)}.png"
        plt.savefig(out, dpi=220); plt.close()
        generated.append(str(out))

        print("Generated files:")
        for g in generated:
            print(" -", g)

if __name__ == "__main__":
    main()
