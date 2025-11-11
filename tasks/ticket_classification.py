from typing import Dict, Any
from .metrics import classify_metrics

SYSTEM = "Classify each ticket as one of: incident, change, problem, maintenance, access, password_reset, etc. Respond with just the label."
PROMPT = """Ticket text:
{text}

Label (incident|change|problem|maintenance|access|password_reset):"""

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    out, stats = model.generate(PROMPT.format(text=ex["text"]), system=SYSTEM)
    pred = (out or "").strip().lower().split()[0]
    gold = (ex.get("label","") or "").lower()
    # compute simple one-example metrics by wrapping
    m = classify_metrics([gold], [pred])
    return {
        "output": out,
        **m,
        **stats
    }
