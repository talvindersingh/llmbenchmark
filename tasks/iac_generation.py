from typing import Dict, Any
from .utils import terraform_validate, arm_template_validate
from .metrics import rouge_l_f, token_f1

SYSTEM = "You are an SRE generating safe, minimal Infrastructure-as-Code."
PROMPT_TF = """Generate Terraform (HCL) that satisfies this spec. Return only HCL.
Spec:
{infra_spec}
"""

PROMPT_ARM = """Generate an Azure ARM (JSON) template for this spec. Return only valid JSON.
Spec:
{infra_spec}
"""

def run_example(model, ex: Dict[str, Any]) -> Dict[str, Any]:
    target = (ex.get("target","terraform") or "terraform").lower()
    if target == "terraform":
        prompt = PROMPT_TF.format(infra_spec=ex["infra_spec"])
    else:
        prompt = PROMPT_ARM.format(infra_spec=ex["infra_spec"])
    out, stats = model.generate(prompt, system=SYSTEM)
    syntax_valid = False
    if target == "terraform":
        syntax_valid = terraform_validate(out)
    else:
        syntax_valid = arm_template_validate(out)
    ref = ex.get("reference","")
    return {
        "output": out,
        "syntax_valid": syntax_valid,
        "validator_success": syntax_valid,
        "rougeL_f": rouge_l_f(ref, out) if ref else None,
        "token_f1": token_f1(ref, out) if ref else None,
        **stats
    }
