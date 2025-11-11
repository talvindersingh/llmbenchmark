from typing import Dict, Any, List
from .metrics import rouge_l_f, token_f1, retrieval_metrics

SYSTEM = "Retrieve relevant docs and create a faithful, concise multi-document summary."
PROMPT = """Query: {query}

Docs (id::text):
{docs}

Return JSON:
{{
  "ranking": ["doc_id1","doc_id2","doc_id3"],
  "summary": "text"
}}
"""

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    docs_formatted = "\n".join([f"{d['id']}::{d['text']}" for d in ex.get("corpus", [])])
    prompt = PROMPT.format(query=ex["query"], docs=docs_formatted)
    out, stats = model.generate(prompt, system=SYSTEM)
    import json
    try:
        obj = json.loads(out)
        ranking = obj.get("ranking", [])
        summary = obj.get("summary", "")
    except Exception:
        ranking, summary = [], out
    gold_docs = ex.get("gold_docs", [])
    retr = retrieval_metrics([gold_docs], [ranking], k=5)
    ref_sum = ex.get("reference_summary","")
    return {
        "output": out,
        **{f"retr_{k}": v for k,v in retr.items()},
        "rougeL_f": rouge_l_f(ref_sum, summary) if ref_sum else None,
        "token_f1": token_f1(ref_sum, summary) if ref_sum else None,
        **stats
    }
