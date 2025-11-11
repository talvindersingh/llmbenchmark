from .utils import rouge_l_f, token_f1, classify_metrics, retrieval_metrics, extract_entities_from_code

def code_doc_entity_precision(ref_code: str, produced_doc: str) -> float:
    ents = set(extract_entities_from_code(ref_code))
    if not ents:
        return 0.0
    hits = sum(1 for e in ents if e in produced_doc)
    return hits / len(ents)

__all__ = ["rouge_l_f", "token_f1", "classify_metrics", "retrieval_metrics", "code_doc_entity_precision"]
