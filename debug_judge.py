"""
Temporary debugging utility to exercise the judge model directly.

Given a benchmark config, a task name, and a results CSV from bench.py,
this script replays the judge scoring so you can see why scores might be
missing (e.g., network failures, malformed prompts, etc.).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

try:
    from ..bench import build_judge_spec, OpenAIJudge, load_jsonl  # type: ignore
except ImportError:
    # Support running as a standalone script by adding package root to sys.path
    import sys
    PACKAGE_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PACKAGE_ROOT))
    from bench import build_judge_spec, OpenAIJudge, load_jsonl  # type: ignore


TASK_DATASETS = {
    "code_generation": "code_generation.jsonl",
    "code_review": "code_review.jsonl",
    "doc_generation": "doc_generation.jsonl",
    "iac_generation": "iac_generation.jsonl",
    "log_summarization": "log_summarization.jsonl",
    "ticket_classification": "ticket_classification.jsonl",
    "kb_qa": "kb_qa.jsonl",
    "doc_retrieval_summarization": "doc_retrieval_sum.jsonl",
}


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_judge(judge_cfg: Dict[str, Any]) -> OpenAIJudge | None:
    if not judge_cfg:
        print("[WARN] No judge_model configured.")
        return None
    if isinstance(judge_cfg, str):
        print("[WARN] Judge is configured as Ollama string; this helper only supports OpenAI judge.")
        return None
    provider = judge_cfg.get("provider", "ollama")
    if provider != "openai":
        print(f"[WARN] Judge provider {provider} is not supported by this helper.")
        return None
    api_env = judge_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_env)
    if not api_key:
        print(f"[ERROR] Environment variable {api_env} not set; cannot call OpenAI judge.")
        return None
    return OpenAIJudge(
        judge_cfg.get("name") or judge_cfg.get("model"),
        api_key,
        base_url=judge_cfg.get("base_url"),
        options=judge_cfg.get("options"),
    )


def main():
    parser = argparse.ArgumentParser(description="Debug judge scoring for a specific task/results file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to bench config.yaml")
    parser.add_argument("--task", type=str, default="code_generation", help="Task name to inspect")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="CSV produced by bench.py containing raw outputs (e.g., results/latest.csv)",
    )
    parser.add_argument("--model", type=str, help="Optional model name filter")
    parser.add_argument("--max", type=int, default=5, help="Max rows to score for quick debugging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    judge = init_judge(cfg.get("judge_model"))
    if not judge:
        print("[ERROR] Judge initialization failed; exiting.")
        return

    data_dir = Path(cfg.get("data_dir", "data"))
    dataset_file = TASK_DATASETS.get(args.task)
    if not dataset_file:
        raise ValueError(f"Unknown task {args.task}")
    ds_path = data_dir / dataset_file
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    examples = {ex["id"]: ex for ex in load_jsonl(str(ds_path))}
    df = pd.read_csv(args.results)
    df = df[df["task"] == args.task]
    if args.model:
        df = df[df["model"] == args.model]
    if df.empty:
        print("[WARN] No matching rows found in results CSV.")
        return

    rows_checked = 0
    scores = []
    for _, row in df.iterrows():
        ex_id = row.get("id")
        example = examples.get(ex_id)
        if not example:
            print(f"[WARN] Example id {ex_id} missing from dataset; skipping.")
            continue
        res = {"output": row.get("output", "")}
        spec = build_judge_spec(args.task, example, res)
        if not spec:
            print(f"[WARN] Could not build rubric for example {ex_id}; skipping.")
            continue
        score = judge.judge_score(spec["rubric"], spec["response"], spec["max_score"])
        if score is None:
            print(f"[WARN] Judge returned None for example {ex_id}")
        else:
            print(f"[INFO] Example {ex_id} -> score {score}/{spec['max_score']}")
            scores.append(score)
        rows_checked += 1
        if rows_checked >= args.max:
            break

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"[INFO] Average score over {len(scores)} examples: {avg_score:.2f}")
    else:
        print("[WARN] No scores computed.")


if __name__ == "__main__":
    main()
