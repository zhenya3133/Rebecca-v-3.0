# Можно заменить на свой API или open-source LLM (например, llama.cpp, openai)
def llm_judge_relevancy(query, doc_text):
    """
    Функция-интерфейс к LLM: возвращает score [0..1] - насколько doc_text релевантен к query.
    Для MVP — мок-оценка по наличию слов, для прод: интегрировать свой prompt.
    """
    prompt = (
        f"Вопрос: {query}\n"
        f"Текст ответа: {doc_text}\n"
        "Оцени РЕЛЕВАНТНОСТЬ ответа по шкале 0 (нет связи) до 1 (идеально):"
    )
    # Вместо следующей строки — вызов своего LLM API
    import random
    return round(random.uniform(0.5, 1.0), 2)  # Мок, всегда ≥0.5


# Пример для future: использовать openai, HF Transformers или local LLM
# def llm_judge_relevancy(query, doc_text): ...
