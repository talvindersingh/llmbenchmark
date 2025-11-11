import os, json, argparse, time, yaml, requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from models import OllamaModel

# Import task modules
from tasks import code_generation, code_review, doc_generation, iac_generation, log_summarization, ticket_classification, kb_qa, doc_retrieval_sum

TASK_MAP = {
    "code_generation": code_generation,
    "code_review": code_review,
    "doc_generation": doc_generation,
    "iac_generation": iac_generation,
    "log_summarization": log_summarization,
    "ticket_classification": ticket_classification,
    "kb_qa": kb_qa,
    "doc_retrieval_summarization": doc_retrieval_sum,
}

class OpenAIJudge:
    def __init__(self, name: str, api_key: str, base_url: str | None = None, options: dict | None = None):
        self.name = name
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.options = options or {}

    def judge_score(self, rubric_prompt: str, response_text: str, maximum: int = 5) -> float | None:
        prompt = f"{rubric_prompt}\n\n<<<RESPONSE START>>>\n{response_text}\n<<<RESPONSE END>>>"
        messages = [
            {"role": "system", "content": "You are a strict grader. Respond with a single integer score."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.name,
            "messages": messages,
            "temperature": self.options.get("temperature", 0.0),
            "max_tokens": self.options.get("max_tokens", 32),
        }
        for key in ("frequency_penalty", "presence_penalty", "top_p"):
            if key in self.options:
                payload[key] = self.options[key]
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
                verify=False,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] OpenAI judge request failed: {e}")
            return None
        data = resp.json()
        text = ""
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            return None
        import re
        m = re.search(r"(\d+)", text or "")
        if not m:
            return None
        score = int(m.group(1))
        return max(0, min(score, maximum))

def build_judge_spec(task_name: str, ex: dict, res: dict) -> dict | None:
    """
    Build rubric prompt + metadata for judge model.
    Returns dict with keys: metric, rubric, response, max_score.
    """
    output_text = res.get("output", "")
    if not output_text:
        return None

    if task_name == "code_generation":
        spec = ex.get("spec")
        if not spec:
            print(f"[WARN] code_generation example {ex.get('id')} missing spec; skipping judge.")
            return None
        language = ex.get("language", "unspecified")
        rubric = (
            "You grade infrastructure/backend code for meeting a CRUD API specification.\n"
            "Score 0-5 based on how completely and correctly the submission satisfies the requirements, "
            "including language/framework conventions, data model, endpoints, persistence, auth, and tests when requested.\n"
            "5 = fully compliant and idiomatic; 0 = irrelevant, incorrect, or dangerously incomplete.\n"
            "Respond with a single integer.\n"
            f"TARGET_LANGUAGE: {language}\n"
            f"SPECIFICATION:\n{spec}"
        )
        return {"metric": "spec_compliance_score", "rubric": rubric, "response": output_text, "max_score": 5}

    if task_name == "code_review":
        gold = ex.get("gold_issues") or []
        snippet = ex.get("snippet")
        if not snippet or not gold:
            return None
        rubric = (
            "You are grading a code review that should surface security/logic issues.\n"
            "Score 0-5 based on how completely and accurately the REVIEW lists the GOLD issues for the SNIPPET.\n"
            "5 = finds all gold issues with precise wording, 0 = misses or hallucinates.\n"
            "Respond with a single integer.\n"
            f"SNIPPET:\n{snippet}\n\n"
            f"GOLD_ISSUES:\n{', '.join(str(g) for g in gold)}"
        )
        return {"metric": "helpfulness_score", "rubric": rubric, "response": output_text, "max_score": 5}

    if task_name == "doc_generation":
        ref = ex.get("reference_doc")
        if not ref:
            return None
        style = ex.get("doc_style", "docstring")
        rubric = (
            "You evaluate generated API documentation.\n"
            "Compare the CANDIDATE doc to the REFERENCE and score coherence & coverage on 0-5 scale.\n"
            "5 = matches reference structure/content closely; 0 = unusable or off-topic.\n"
            "Return one integer.\n"
            f"STYLE_REQUEST: {style}\n\nREFERENCE_DOC:\n{ref}"
        )
        return {"metric": "coherence_score", "rubric": rubric, "response": output_text, "max_score": 5}

    if task_name == "log_summarization":
        ref = ex.get("reference_summary")
        log_text = ex.get("log")
        if not ref or not log_text:
            return None
        rubric = (
            "You assess an SRE incident summary.\n"
            "Given LOG context and a trusted REFERENCE summary, score the SUMMARY's factual faithfulness (0-5).\n"
            "5 = entirely consistent with reference and grounded in log; 0 = incorrect or hallucinated.\n"
            "Answer with a single integer.\n"
            f"LOG:\n{log_text}\n\nREFERENCE_SUMMARY:\n{ref}"
        )
        return {"metric": "faithfulness_score", "rubric": rubric, "response": output_text, "max_score": 5}

    if task_name == "kb_qa":
        question = ex.get("question")
        context = ex.get("context")
        answer = ex.get("answer")
        if not (question and context and answer):
            return None
        rubric = (
            "You grade knowledge-base answers.\n"
            "Score 0-5 for correctness and grounding in CONTEXT.\n"
            "Consider the EXPECTED_ANSWER as authoritative. Penalize unsupported guesses.\n"
            "Return integer only.\n"
            f"QUESTION: {question}\nEXPECTED_ANSWER: {answer}\n\nCONTEXT:\n{context}"
        )
        return {"metric": "groundedness_score", "rubric": rubric, "response": output_text, "max_score": 5}

    if task_name == "doc_retrieval_summarization":
        query = ex.get("query")
        ref_sum = ex.get("reference_summary")
        gold_docs = ex.get("gold_docs")
        if not (query and ref_sum and gold_docs):
            return None
        rubric = (
            "You evaluate a retrieved-document summary.\n"
            "Score 0-5 for how well the SUMMARY answers the QUERY while staying faithful to the REFERENCE summary and GOLD documents.\n"
            "5 = accurate, comprehensive, grounded; 0 = incorrect/off-topic.\n"
            "Reply with integer only.\n"
            f"QUERY: {query}\nREFERENCE_SUMMARY:\n{ref_sum}\nGOLD_DOC_IDS: {', '.join(str(g) for g in gold_docs)}"
        )
        return {"metric": "summary_faithfulness_score", "rubric": rubric, "response": output_text, "max_score": 5}

    if task_name == "iac_generation":
        spec = ex.get("infra_spec")
        target = ex.get("target", "terraform")
        if not spec:
            return None
        rubric = (
            "You audit Infrastructure-as-Code for compliance, safety, and completeness.\n"
            "Score 0-5 based on whether the template fulfills the SPEC, uses secure defaults, "
            "and adheres to best practices for the declared target format.\n"
            "5 = fully correct, secure, and deployable; 0 = unusable or unsafe.\n"
            "Respond with a single integer.\n"
            f"TARGET_FORMAT: {target}\n"
            f"SPEC:\n{spec}"
        )
        return {"metric": "infra_compliance_score", "rubric": rubric, "response": output_text, "max_score": 5}

    return None

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = [OllamaModel(m["name"], m.get("options")) for m in cfg["models"]]
    judge_cfg = cfg.get("judge_model")
    judge_model = None
    if judge_cfg:
        if isinstance(judge_cfg, str):
            judge_model = OllamaModel(judge_cfg)
        elif isinstance(judge_cfg, dict):
            provider = judge_cfg.get("provider", "ollama")
            if provider == "openai":
                api_env = judge_cfg.get("api_key_env", "OPENAI_API_KEY")
                api_key = os.environ.get(api_env)
                if not api_key:
                    print(f"[WARN] judge_model provider=openai but env var {api_env} is not set; skipping judge.")
                else:
                    judge_model = OpenAIJudge(
                        judge_cfg.get("name") or judge_cfg.get("model"),
                        api_key,
                        base_url=judge_cfg.get("base_url"),
                        options=judge_cfg.get("options"),
                    )
            else:
                judge_model = OllamaModel(judge_cfg.get("name") or judge_cfg.get("model"), judge_cfg.get("options"))
    data_dir = Path(cfg.get("data_dir","data"))
    out_dir = Path(cfg.get("output_dir","results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = cfg["tasks"]
    rows = []
    ts = time.strftime("%Y%m%d_%H%M%S")

    for task_name in tasks:
        task_mod = TASK_MAP[task_name]
        file_map = {
            "code_generation": "code_generation.jsonl",
            "code_review": "code_review.jsonl",
            "doc_generation": "doc_generation.jsonl",
            "iac_generation": "iac_generation.jsonl",
            "log_summarization": "log_summarization.jsonl",
            "ticket_classification": "ticket_classification.jsonl",
            "kb_qa": "kb_qa.jsonl",
            "doc_retrieval_summarization": "doc_retrieval_sum.jsonl",
        }
        ds_path = data_dir / file_map[task_name]
        if not ds_path.exists():
            print(f"[WARN] Missing dataset for {task_name}: {ds_path}")
            continue
        data = list(load_jsonl(str(ds_path)))

        for model in models:
            for ex in tqdm(data, desc=f"{task_name} :: {model.name}"):
                try:
                    res = task_mod.run_example(model, ex)
                except Exception as e:
                    res = {"error": str(e)}
                judge_scores = {}
                if judge_model:
                    spec = build_judge_spec(task_name, ex, res)
                    if spec:
                        score = judge_model.judge_score(spec["rubric"], spec["response"], spec["max_score"])
                        if score is None:
                            print(f"[WARN] Judge returned None for {task_name} example {ex.get('id')} with model {model.name}.")
                        else:
                            judge_scores[spec["metric"]] = score
                row = {
                    "task": task_name,
                    "id": ex.get("id"),
                    "model": model.name,
                    **{k: v for k, v in res.items() if k != "output"},
                    **judge_scores,
                    "output": res.get("output","")
                }
                rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = out_dir / f"{ts}.csv"
    df.to_csv(out_csv, index=False)
    latest = out_dir / "latest.csv"
    df.to_csv(latest, index=False)

    if not df.empty:
        grouped = df.groupby("model")
        summary = grouped.size().rename("sample_count").to_frame()
        agg_candidates = {
            "compile_success": "mean",
            "static_warnings": "mean",
            "rougeL_f_vs_spec": "mean",
            "token_f1_vs_spec": "mean",
            "spec_compliance_score": "mean",
            "eval_count": "mean",
            "eval_duration_ms": "mean",
            "prompt_eval_count": "mean",
            "prompt_eval_duration_ms": "mean",
            "total_duration_ms": "mean",
            "wall_time_ms": "mean",
        }
        for col, agg in agg_candidates.items():
            if col in df.columns:
                summary[col] = grouped[col].mean()
        if "compile_success" in df.columns:
            summary["compile_success_rate_pct"] = summary["compile_success"] * 100
        summary = summary.reset_index()
        summary_csv = out_dir / f"{ts}_summary.csv"
        summary.to_csv(summary_csv, index=False)
        latest_summary = out_dir / "latest_summary.csv"
        summary.to_csv(latest_summary, index=False)
        print(f"Wrote summary: {summary_csv}")
        print(f"Also wrote summary: {latest_summary}")

    print(f"Wrote: {out_csv}")
    print(f"Also wrote: {latest}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
