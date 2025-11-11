from typing import Dict, Any
from .metrics import rouge_l_f, token_f1, code_doc_entity_precision

SYSTEM = "You produce concise, accurate API documentation in Markdown."
PROMPT = """Create API documentation for the following code in the requested style.
Style: {doc_style}

```
{code}
```

Return only the documentation.
"""

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PROMPT.format(doc_style=ex.get("doc_style","docstring"), code=ex["code"])
    out, stats = model.generate(prompt, system=SYSTEM)
    ref = ex.get("reference_doc","")
    return {
        "output": out,
        "rougeL_f": rouge_l_f(ref, out) if ref else None,
        "token_f1": token_f1(ref, out) if ref else None,
        "entity_precision": code_doc_entity_precision(ex.get("code",""), out),
        **stats
    }
