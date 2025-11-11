import re
from typing import Dict, Any
from .metrics import token_f1

SYSTEM = "Answer concisely using only the provided KB context. Cite supporting lines in parentheses."
PROMPT = """Question: {question}

KB Context:
\"\"\"
{context}
\"\"\"

Answer briefly and cite line numbers in () where relevant.
"""

def exact_match(a: str, b: str) -> float:
    return 1.0 if (a or "").strip().lower() == (b or "").strip().lower() else 0.0

def groundedness_score(text: str) -> float:
    # naive: consider presence of citations like (12) or (3-4) as proxy for grounding
    return 1.0 if re.search(r'\(\d+(\-\d+)?\)', text or "") else 0.0

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PROMPT.format(question=ex["question"], context=ex.get("context",""))
    out, stats = model.generate(prompt, system=SYSTEM)
    em = exact_match(out, ex.get("answer",""))
    f1 = token_f1(ex.get("answer",""), out)
    grd = groundedness_score(out)
    return {
        "output": out,
        "exact_match": em,
        "token_f1": f1,
        "groundedness": grd,
        **stats
    }
