import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

TASK_METRIC_SPECS = {
    "code_generation": [
        ("compile_success", "Code Generation: Compile Success Rate"),
        ("static_warnings", "Code Generation: Bandit Warnings (lower is better)"),
        ("rougeL_f_vs_spec", "Code Generation: ROUGE-L vs Spec"),
        ("token_f1_vs_spec", "Code Generation: Token F1 vs Spec"),
    ],
    "code_review": [
        ("precision", "Code Review: Precision"),
        ("recall", "Code Review: Recall"),
        ("f1", "Code Review: F1"),
        ("helpfulness_score", "Code Review: Judge Helpfulness Score"),
    ],
    "doc_generation": [
        ("rougeL_f", "Document Generation: ROUGE-L F"),
        ("token_f1", "Document Generation: Token F1"),
        ("entity_precision", "Document Generation: Entity Precision"),
        ("coherence_score", "Document Generation: Judge Coherence Score"),
    ],
    "iac_generation": [
        ("syntax_valid", "IaC Generation: Syntax Valid Rate"),
        ("validator_success", "IaC Generation: Validator Success Rate"),
        ("rougeL_f", "IaC Generation: ROUGE-L F"),
        ("token_f1", "IaC Generation: Token F1"),
    ],
    "log_summarization": [
        ("rougeL_f", "Log Summarization: ROUGE-L F"),
        ("token_f1", "Log Summarization: Token F1"),
        ("compression_ratio", "Log Summarization: Compression Ratio"),
        ("faithfulness_score_0to5", "Log Summarization: QA Faithfulness Score (0-5)"),
        ("faithfulness_score", "Log Summarization: Judge Faithfulness Score"),
    ],
    "ticket_classification": [
        ("accuracy", "Ticket Classification: Accuracy"),
        ("macro_f1", "Ticket Classification: Macro F1"),
    ],
    "kb_qa": [
        ("exact_match", "KB Q&A: Exact Match"),
        ("token_f1", "KB Q&A: Token F1"),
        ("groundedness", "KB Q&A: Citation Groundedness"),
        ("groundedness_score", "KB Q&A: Judge Groundedness Score"),
    ],
    "doc_retrieval_summarization": [
        ("retr_recall@k", "Doc Retrieval & Summarization: Recall@k"),
        ("retr_mrr@k", "Doc Retrieval & Summarization: MRR@k"),
        ("retr_ndcg@k", "Doc Retrieval & Summarization: nDCG@k"),
        ("rougeL_f", "Doc Retrieval & Summarization: ROUGE-L F"),
        ("token_f1", "Doc Retrieval & Summarization: Token F1"),
        ("summary_faithfulness_score", "Doc Retrieval & Summarization: Judge Faithfulness Score"),
    ],
}

NEGATIVE_IS_MISSING = {"static_warnings"}

def save_bar(df: pd.DataFrame, metric: str, title: str, out_path: str) -> bool:
    grouped = df.groupby("model")[metric].mean().dropna()
    if grouped.empty:
        return False
    ax = grouped.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True

def slugify_filename(task: str, metric: str) -> str:
    base = f"{task}_{metric}"
    base = base.replace("@", "at")
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_").lower()
    return f"{base}.png"

def main(in_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    charts = []

    if not {"task", "model"}.issubset(df.columns):
        print("Missing task/model columns; cannot generate charts.")
        return

    for task, specs in TASK_METRIC_SPECS.items():
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue
        for metric, title in specs:
            if metric not in task_df.columns:
                continue
            metric_series = pd.to_numeric(task_df[metric], errors="coerce")
            if metric in NEGATIVE_IS_MISSING:
                metric_series = metric_series.where(metric_series >= 0)
            metric_series = metric_series.dropna()
            if metric_series.empty:
                continue
            plot_df = task_df.loc[metric_series.index].copy()
            plot_df[metric] = metric_series
            filename = slugify_filename(task, metric)
            out_path = os.path.join(out_dir, filename)
            if save_bar(plot_df, metric, title, out_path):
                charts.append(out_path)

    print("Charts saved:", charts)

if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "results/latest.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "charts"
    main(in_csv, out_dir)
