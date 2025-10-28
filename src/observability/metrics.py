"""Retrieval quality metrics for observability and regression tracking."""

import datetime


def coverage_at_k(retrieved, ground_truth, k=5):
    topk = set(retrieved[:k])
    gt = set(ground_truth)
    if not gt:
        return 0.0
    return len(topk & gt) / min(len(gt), k)


def contradiction_rate(contexts):
    contradictions = [c for c in contexts if "contradiction" in str(c) or "error" in str(c)]
    return len(contradictions) / max(1, len(contexts))


def token_efficiency(retrieved, token_budget):
    if token_budget <= 0:
        return 0.0
    tokens = sum(len(str(c).split()) for c in retrieved)
    return tokens / token_budget


def drift_score(val_list, reference, window_days=7):
    recent = [item["value"] for item in val_list if (datetime.datetime.now() - item["ts"]).days <= window_days]
    if not recent:
        return 0.0
    avg = sum(recent) / len(recent)
    return abs(avg - reference) / (abs(reference) + 1e-8)


def privacy_violation_rate(records, policy_engine):
    total = 0
    violations = 0
    for record in records:
        node = record.get("node")
        if node is None:
            continue
        total += 1
        if not policy_engine.allowed("audit", node, "read"):
            violations += 1
    if total == 0:
        return 0.0
    return violations / total
