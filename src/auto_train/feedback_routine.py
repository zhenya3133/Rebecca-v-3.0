def update_scores_based_on_feedback(retrieval_log, feedback):
    """
    retrieval_log: [{'query':..., 'result':..., 'score':...}]
    feedback: [{'query':..., 'result':..., 'good':bool}]
    """
    for fb in feedback:
        for record in retrieval_log:
            if record.get("query") == fb.get("query") and record.get("result") == fb.get("result"):
                if fb.get("good"):
                    record["score"] = record.get("score", 0.0) + 0.05
                else:
                    record["score"] = record.get("score", 0.0) - 0.1
    # можно расширить: авто-переобучение моделей или фильтров


# Пример вызова, интегрируй в nightly eval при наличии ручного/авто-отзыва
