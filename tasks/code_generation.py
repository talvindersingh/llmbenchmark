import json
from typing import Dict, Any
from .utils import py_compile_check, java_compile_check, nodejs_check, bandit_warnings
from .metrics import rouge_l_f, token_f1

SYSTEM = "You are a senior backend engineer. Generate minimal, secure, idiomatic code only."

PROMPT = """You are given a CRUD API specification.
Language: {language}
Spec:
{spec}

Return ONLY the code, no explanations.
"""

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PROMPT.format(language=ex.get("language","python"), spec=ex["spec"])
    out, stats = model.generate(prompt, system=SYSTEM)
    language = ex.get("language","python").lower()
    compile_ok = False
    static_warn = None
    if language == "python":
        compile_ok = py_compile_check(out)
        static_warn = bandit_warnings(out)
    elif language == "java":
        compile_ok = java_compile_check(out)
    elif language in {"javascript", "nodejs", "node", "js"}:
        compile_ok = nodejs_check(out)
    return {
        "output": out,
        "compile_success": compile_ok,
        "static_warnings": static_warn,
        "rougeL_f_vs_spec": rouge_l_f(ex.get("spec",""), out),
        "token_f1_vs_spec": token_f1(ex.get("spec",""), out),
        **stats
    }
