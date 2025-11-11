from typing import Dict, Any, List
from .metrics import rouge_l_f, token_f1

SYSTEM = "You are a site reliability engineer summarizing logs into a concise root cause."
PROMPT = """Summarize the following log into a root-cause summary (2-4 sentences).

LOG:
```
{log}
```
"""

def qa_faithfulness_score(model, summary: str, qa_checks: List[dict]) -> float | None:
    if not qa_checks:
        return None
    rubric = """You are a strict grader. Score 0-5: does the SUMMARY answer the QUESTION correctly based ONLY on the FACT?
Return just the integer.
"""
    scores = []
    for item in qa_checks:
        q = item.get("q","")
        a = item.get("a","")
        prompt = f"{rubric}\nQUESTION: {q}\nFACT: {a}\nSUMMARY: {summary}"
        try:
            s, _ = model.generate(prompt)
            import re
            m = re.search(r'(\d+)', s or "")
            if m:
                scores.append(max(0, min(int(m.group(1)), 5)))
        except Exception:
            continue
    if not scores:
        return None
    return sum(scores)/len(scores)

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PROMPT.format(log=ex["log"])
    out, stats = model.generate(prompt, system=SYSTEM)
    ref = ex.get("reference_summary","")
    qa = ex.get("qa_checks", [])
    faith = qa_faithfulness_score(model, out, qa)
    comp_ratio = (len(out.split()) / max(1, len(ex["log"].split())))
    return {
        "output": out,
        "rougeL_f": rouge_l_f(ref, out) if ref else None,
        "token_f1": token_f1(ref, out) if ref else None,
        "compression_ratio": comp_ratio,
        "faithfulness_score_0to5": faith,
        **stats
    }
