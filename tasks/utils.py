import os, re, json, tempfile, subprocess, textwrap, math
from typing import List, Dict, Any

def safe_write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def which(cmd: str) -> bool:
    from shutil import which as _which
    return _which(cmd) is not None

def run_subprocess(cmd: list[str], cwd: str | None = None, timeout: int = 60):
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return -1, "", str(e)

def extract_entities_from_code(code: str) -> List[str]:
    """
    Very simple heuristic for API entities: function names and HTTP verbs.
    """
    ents = set()
    for m in re.finditer(r"def\s+([a-zA-Z_]\w*)\s*\(", code):
        ents.add(m.group(1))
    for verb in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
        if verb in code.upper():
            ents.add(verb)
    return sorted(ents)

def lcs(a: list[str], b: list[str]) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l_f(ref: str, pred: str) -> float:
    a = ref.split()
    b = pred.split()
    if not a or not b:
        return 0.0
    l = lcs(a, b)
    prec = l / len(b)
    rec = l / len(a)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def token_f1(ref: str, pred: str) -> float:
    a = ref.split()
    b = pred.split()
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prec = common / len(b)
    rec = common / len(a)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def classify_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    from collections import defaultdict, Counter
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    # per-class
    per = {}
    for lab in labels:
        tp = cm[lab][lab]
        fp = sum(cm[other][lab] for other in labels if other != lab)
        fn = sum(cm[lab][other] for other in labels if other != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1}
    acc = sum(cm[l][l] for l in labels) / len(y_true) if y_true else 0.0
    macro_f1 = sum(per[l]["f1"] for l in labels) / len(labels) if labels else 0.0
    # convert cm to dict of dict
    cm_out = {r: dict(c) for r, c in cm.items()}
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per, "confusion_matrix": cm_out}

def retrieval_metrics(gold_docs: List[str], ranked_lists: List[List[str]], k: int = 5):
    """
    ranked_lists: list of ranked doc ids per query (we assume same order as gold_docs entries).
    For this harness we compute simple versions of Recall@k, MRR@k, nDCG@k.
    """
    def dcg(rel):
        return sum((rel[i] / math.log2(i+2) for i in range(len(rel))))
    recalls, mrrs, ndcgs = [], [], []
    for gold, ranked in zip(gold_docs, ranked_lists):
        rel = [1 if d == gold or (isinstance(gold, list) and d in gold) else 0 for d in ranked[:k]]
        recall = sum(rel) / (1 if isinstance(gold, str) else len(gold))
        # MRR
        rr = 0.0
        for i, r in enumerate(rel, 1):
            if r == 1:
                rr = 1.0 / i
                break
        # nDCG
        ideal = sorted(rel, reverse=True)
        ndcg = dcg(rel) / (dcg(ideal) or 1.0)
        recalls.append(recall)
        mrrs.append(rr)
        ndcgs.append(ndcg)
    n = len(gold_docs) or 1
    return {"recall@k": sum(recalls)/n, "mrr@k": sum(mrrs)/n, "ndcg@k": sum(ndcgs)/n}

def py_compile_check(code: str) -> bool:
    import tempfile, py_compile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            py_compile.compile(path, doraise=True)
            return True
        except Exception:
            return False

def java_compile_check(code: str) -> bool:
    if not which("javac"):
        return False
    import tempfile, os, re, subprocess
    with tempfile.TemporaryDirectory() as td:
        # naive: derive class name or use Main.java
        class_name = "Main"
        m = re.search(r"class\s+([A-Z][A-Za-z0-9_]*)", code)
        if m:
            class_name = m.group(1)
        path = os.path.join(td, f"{class_name}.java")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        rc, out, err = run_subprocess(["javac", path], cwd=td)
        return rc == 0

def nodejs_check(code: str) -> bool:
    if not which("node"):
        return False
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "index.js")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        rc, out, err = run_subprocess(["node", "--check", path], cwd=td)
        return rc == 0

def bandit_warnings(py_code: str) -> int:
    if not which("bandit"):
        return -1
    import tempfile, os, json
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "code.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(py_code)
        rc, out, err = run_subprocess(["bandit", "-q", "-f", "json", p], cwd=td)
        try:
            data = json.loads(out or "{}")
            return len(data.get("results", []))
        except Exception:
            return -1

def _validate_arm_resource(resource: dict) -> bool:
    if not isinstance(resource, dict):
        return False
    required_fields = ["type", "name"]
    for field in required_fields:
        value = resource.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
    api_version = resource.get("apiVersion")
    if api_version is not None and (not isinstance(api_version, str) or not api_version.strip()):
        return False
    properties = resource.get("properties")
    if properties is not None and not isinstance(properties, dict):
        return False
    if "resources" in resource:
        nested = resource["resources"]
        if not isinstance(nested, list) or not all(_validate_arm_resource(r) for r in nested):
            return False
    return True

def arm_template_validate(arm_text: str) -> bool:
    """
    Minimal Azure ARM validation that ensures the template has the required top-level
    fields and that resource blocks include key metadata fields.
    """
    try:
        template = json.loads(arm_text)
    except Exception:
        return False
    if not isinstance(template, dict):
        return False
    if not isinstance(template.get("$schema"), str):
        return False
    content_version = template.get("contentVersion")
    if not isinstance(content_version, str) or not re.match(r"^\d+\.\d+\.\d+\.\d+$", content_version):
        return False
    resources = template.get("resources")
    if not isinstance(resources, list) or not resources:
        return False
    if not all(_validate_arm_resource(res) for res in resources):
        return False
    parameters = template.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, dict):
            return False
        for param_name, param_value in parameters.items():
            if not isinstance(param_name, str) or not param_name.strip():
                return False
            if not isinstance(param_value, dict):
                return False
            if not isinstance(param_value.get("type"), str):
                return False
    variables = template.get("variables")
    if variables is not None and not isinstance(variables, dict):
        return False
    outputs = template.get("outputs")
    if outputs is not None and not isinstance(outputs, dict):
        return False
    return True

def terraform_validate(tf_text: str) -> bool:
    if not which("terraform"):
        return False
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "main.tf")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tf_text)
        rc_init, out_init, err_init = run_subprocess(["terraform", "init", "-backend=false"], cwd=td, timeout=120)
        rc_val, out_val, err_val = run_subprocess(["terraform", "validate"], cwd=td, timeout=120)
        return rc_init == 0 and rc_val == 0
