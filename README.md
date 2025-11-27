# Benchmark: Enterprise IT Scenarios (LLM/SLM)

This repository is a **self-contained harness** to benchmark multiple language models (running locally via **Ollama**) on enterprise IT scenarios:
- Software Development: Code Generation, Code Review, Document Generation
- IT Operations: Infrastructure-as-Code (IaC), Log Summarization, Ticket Classification
- Enterprise Knowledge Management: KB Q&A, Document Retrieval & Summarization

It includes:
- Reproducible **metrics** appropriate to each scenario
- A **task runner** with clean abstractions
- **Model adapters** for Ollama
- **Results aggregation** and **comparison charts**

> You can expand or customize any task/dataset without changing the core harness.

---
*Flow diagram*
<img width="627" height="749" alt="image" src="https://github.com/user-attachments/assets/87b196ad-7ff5-43d6-8f3a-0ca4ef435f6e" />


## Quick Start

1. Install dependencies (Python 3.10+ recommended)
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure **Ollama** is running and your models are pulled, e.g.:
   ```bash
   ollama run llama3.1
   ollama run qwen2.5
   ```

3. Edit `config.yaml` to list the models and tasks you want to run.

4. Put your datasets under `data/` using the provided schema examples.

5. Run the benchmark:
   ```bash
   python bench.py --config config.yaml
   ```

6. Generate comparison charts:
   ```bash
   python analysis/make_charts.py results/latest.csv charts/
   ```

---

## Datasets (schema)

All tasks consume **JSONL** files with one object per example. Minimal schemas are shown below; you can add fields as needed.

### 1) Code Generation (`data/code_generation.jsonl`)
```json
{{
  "id": "ex1",
  "language": "python|java",
  "spec": "Generate CRUD for 'User' with fields id:int, name:string",
  "tests": ["optional path to unit tests, or inline tests"],
  "constraints": "e.g., PEP8, use FastAPI"
}}
```
**Metrics:** compile_success, unit_tests_pass@1, static_warnings, time_ms, tokens.

### 2) Code Review (`data/code_review.jsonl`)
```json
{{
  "id": "ex1",
  "language": "python|java",
  "snippet": "def foo(x): ...",
  "gold_issues": ["SQL injection", "Null pointer", "Off-by-one"]
}}
```
**Metrics:** precision, recall, f1 (issue-level), helpfulness_score (LLM judge, optional).

### 3) Document Generation (`data/doc_generation.jsonl`)
```json
{{
  "id": "ex1",
  "code": "def create_user(...): ...",
  "doc_style": "API docstring or Markdown",
  "reference_doc": "Ground truth doc for overlap metrics"
}}
```
**Metrics:** rougeL_f, token_f1, entity_precision (APIs/entities detected), coherence_score (LLM judge, optional).

### 4) IaC Generation (`data/iac_generation.jsonl`)
```json
{{
  "id": "ex1",
  "target": "terraform|arm",
  "infra_spec": "Create an S3 bucket with versioning",
  "policies": ["optional OPA/Rego policies"],
  "reference": "optional ground truth template"
}}
```
**Metrics:** syntax_valid, validator_success (e.g., `terraform validate`), policy_compliant, token_f1 vs reference (if provided).

### 5) Log Summarization (`data/log_summarization.jsonl`)
```json
{{
  "id": "ex1",
  "log": "Large log text...",
  "reference_summary": "Root cause summary",
  "qa_checks": [{{"q":"What was the root cause?","a":"OOM on node-3"}}]
}}
```
**Metrics:** rougeL_f, token_f1, compression_ratio, faithfulness_score (QA-based judge).

### 6) Ticket Classification (`data/ticket_classification.jsonl`)
```json
{{
  "id": "ex1",
  "text": "Server reboot scheduled ...",
  "label": "change|incident|problem"
}}
```
**Metrics:** accuracy, macro_f1, per_class_precision/recall/f1, confusion_matrix.

### 7) KB Q&A (`data/kb_qa.jsonl`)
```json
{{
  "id": "ex1",
  "question": "How to reset VPN?",
  "context": "Relevant KB extract(s)",
  "answer": "Expected short answer"
}}
```
**Metrics:** exact_match, token_f1, groundedness_score (judge cites lines).

### 8) Document Retrieval & Summarization (`data/doc_retrieval_sum.jsonl`)
```json
{{
  "id": "ex1",
  "query": "Decommission steps for service X",
  "corpus": [{{"id":"d1","text":"..."}}],
  "gold_docs": ["d3","d9"],
  "reference_summary": "Expected multi-doc summary"
}}
```
**Metrics:** recall@k, mrr@k, ndcg@k, rougeL_f and token_f1 for summary.

---

## Results

- The harness writes `results/{{timestamp}}.csv` and also updates/overwrites `results/latest.csv` for convenience.
- Each row: one (task, example, model) with raw outputs, metrics, and timings.

---

## Advanced/Optional Checks

- **Static analysis:** `bandit` (Python), `semgrep` (multi-language) for warnings count.
- **Compilation:** `py_compile` (Python), `javac` (Java).
- **IaC validation:** `terraform validate` (Terraform).
- **LLM-as-a-judge:** A second model in `config.yaml` can grade faithfulness/helpfulness with explicit rubrics.

> If a tool is missing on your machine, the harness skips that check and logs a warning.

---

## Citation & Ethics

Report: datasets, prompt templates, hyperparameters, judge rubric, and model versions (Ollama tags).
Include latency and token throughput from Ollama response fields where available.

