import hashlib

try:
    from configuration import is_offline_mode
except ImportError:
    import os
    def is_offline_mode() -> bool:
        return (
            os.environ.get("REBECCA_OFFLINE_MODE", "").lower() in ("1", "true", "yes", "on") or
            os.environ.get("REBECCA_TEST_MODE", "").lower() in ("1", "true", "yes", "on")
        )


def llm_judge_relevancy(query, doc_text):
    """
    Функция-интерфейс к LLM: возвращает score [0..1] - насколько doc_text релевантен к query.
    
    В offline mode использует детерминированную оценку на основе пересечения слов.
    В online mode может использовать реальный LLM API.
    """
    if is_offline_mode():
        return _deterministic_relevancy_score(query, doc_text)
    
    # В online mode можно интегрировать реальный LLM API
    # Пока используем детерминированный метод
    return _deterministic_relevancy_score(query, doc_text)


def _deterministic_relevancy_score(query: str, doc_text: str) -> float:
    """
    Детерминированная оценка релевантности на основе пересечения слов.
    Возвращает score в диапазоне [0.3, 1.0] для обеспечения минимальной релевантности.
    """
    query_lower = query.lower()
    doc_lower = doc_text.lower()
    
    # Простой подсчет пересечения слов
    query_words = set(query_lower.split())
    doc_words = set(doc_lower.split())
    
    if not query_words:
        return 0.5
    
    # Пересечение слов
    intersection = query_words & doc_words
    overlap_ratio = len(intersection) / len(query_words)
    
    # Нормализуем в диапазон [0.3, 1.0]
    score = 0.3 + (overlap_ratio * 0.7)
    
    # Добавляем небольшую детерминированную компоненту на основе хеша
    # для различия между документами с одинаковым overlap
    hash_val = int(hashlib.md5((query + doc_text).encode()).hexdigest()[:8], 16)
    hash_component = (hash_val % 100) / 1000  # 0.000 to 0.099
    
    return round(min(1.0, score + hash_component), 2)


# Пример для future: использовать openai, HF Transformers или local LLM
# def llm_judge_relevancy(query, doc_text): ...
