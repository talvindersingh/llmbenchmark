import re, json
from typing import Dict, Any, List

SYSTEM = "You are a strict security code reviewer. Respond with a JSON list of issues."
PROMPT = """Review the following {language} function and list SECURITY or LOGIC issues.
Return a *JSON array of strings* (no commentary).

Snippet:
```
{snippet}
```
"""

def normalize(items: List[str]) -> List[str]:
    return [re.sub(r'\W+', ' ', s).lower().strip() for s in items]

def f1_list(gold: List[str], pred: List[str]) -> Dict[str, float]:
    g = set(normalize(gold))
    p = set(normalize(pred))
    tp = len(g & p)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PROMPT.format(language=ex.get("language","python"), snippet=ex["snippet"])
    out, stats = model.generate(prompt, system=SYSTEM)
    try:
        issues = json.loads(out)
        if not isinstance(issues, list):
            raise ValueError("Not a list")
        issues = [str(x) for x in issues]
    except Exception:
        issues = []
    prf = f1_list(ex.get("gold_issues", []), issues)
    return {
        "output": out,
        **prf,
        **stats
    }
