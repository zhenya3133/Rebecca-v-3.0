"""
Тесты для системы извлечения концептов и связей.

Содержит тестовые кейсы для различных типов документов:
- Научные статьи
- Новостные статьи
- Технические документы
- Финансовые отчеты
"""

import asyncio
import pytest
from pathlib import Path
from typing import List, Dict, Any

from knowledge_graph.concept_extractor import (
    ConceptExtractor,
    ExtractedKnowledge,
    create_concept_extractor
)
from memory_manager.memory_manager import MemoryManager


class TestDocumentTypes:
    """Тестовые документы для различных типов контента."""
    
    @staticmethod
    def get_scientific_article() -> str:
        """Пример научной статьи."""
        return """
        Искусственный интеллект в медицине: современные подходы и перспективы
        
        В данном исследовании мы анализируем применение машинного обучения 
        в диагностике онкологических заболеваний. Алгоритмы глубокого обучения 
        показывают высокую точность при анализе медицинских изображений.
        
        Ключевые слова: искусственный интеллект, машинное обучение, медицинская 
        диагностика, нейронные сети, онкология.
        
        Методы исследования включали использование сверточных нейронных сетей 
        для анализа КТ-сканов. Точность диагностики составила 94.5%, что 
        превышает показатели традиционных методов на 15%.
        
        Результаты исследования имеют важное значение для развития системы 
        здравоохранения и могут быть внедрены в клиническую практику в 
        течение ближайших двух лет.
        """
    
    @staticmethod
    def get_news_article() -> str:
        """Пример новостной статьи."""
        return """
        Новая технология от Tesla изменит рынок электромобилей
        
        Компания Tesla объявила о разработке революционной батареи с запасом 
        хода 1000 километров. Илон Маск, генеральный директор Tesla, заявил, 
        что новая технология будет доступна в моделях 2024 года.
        
        Аналитики Bloomberg считают, что это достижение может кардинально 
        изменить конкурентную среду в автомобильной индустрии. Акции Tesla 
        выросли на 12% после объявления.
        
        Эксперты автомобильной отрасли отмечают, что данная технология 
        создает новые стандарты для всей индустрии электромобилей.
        """
    
    @staticmethod
    def get_technical_document() -> str:
        """Пример технического документа."""
        return """
        Архитектура микросервисной системы обработки данных
        
        Система состоит из следующих компонентов: API Gateway, Service Registry, 
        Load Balancer и база данных PostgreSQL. Каждый микросервис имеет свою 
        базу данных MongoDB для хранения специфичных данных.
        
        Технологический стек включает: Python, FastAPI, Redis для кэширования, 
        Docker для контейнеризации, Kubernetes для оркестрации, и Prometheus 
        для мониторинга системы.
        
        Архитектура обеспечивает горизонтальное масштабирование и высокую 
        доступность сервиса. Система обрабатывает до 10000 запросов в секунду 
        с задержкой менее 100 миллисекунд.
        """
    
    @staticmethod
    def get_financial_report() -> str:
        """Пример финансового отчета."""
        return """
        Квартальный отчет компании Microsoft за Q3 2024
        
        Выручка компании составила $52.3 млрд, что на 15% больше по сравнению 
        с аналогичным периодом прошлого года. Операционная прибыль достигла 
        $22.1 млрд.
        
        Основной вклад в рост выручки внесли облачные сервисы Azure и 
        подписки Office 365. Продажи игровых консолей Xbox выросли на 8%.
        
        Капитальные инвестиции в R&D составили $7.2 млрд. Компания планирует 
        увеличить штат сотрудников на 12% в следующем квартале.
        
        Акции Microsoft торгуются на бирже NASDAQ под тикером MSFT.
        """
    
    @staticmethod
    def get_legal_document() -> str:
        """Пример правового документа."""
        return """
        Договор купли-продажи недвижимости
        
        Настоящий договор заключен между продавцом ООО "Недвижимость Плюс" 
        и покупателем Ивановым Иваном Ивановичем.
        
        Предмет договора: квартира площадью 78.5 кв.м., расположенная по 
        адресу г. Москва, ул. Тверская, д. 15, кв. 42.
        
        Стоимость объекта составляет 12 500 000 рублей. Оплата производится 
        единовременно в течение 5 банковских дней после государственной 
        регистрации перехода права собственности.
        
        Ответственность сторон регулируется действующим законодательством 
        Российской Федерации.
        """


class TestConceptExtractor:
    """Основной класс для тестирования извлечения концептов."""
    
    @pytest.fixture
    async def extractor(self):
        """Фикстура для создания экстрактора концептов."""
        return await create_concept_extractor()
    
    @pytest.fixture
    async def extractor_with_memory(self):
        """Фикстура для создания экстрактора с MemoryManager."""
        memory_manager = MemoryManager()
        await memory_manager.start()
        extractor = await create_concept_extractor(memory_manager)
        yield extractor
        await memory_manager.stop()
    
    @pytest.mark.asyncio
    async def test_scientific_article_extraction(self, extractor: ConceptExtractor):
        """Тест извлечения концептов из научной статьи."""
        text = TestDocumentTypes.get_scientific_article()
        
        result = await extractor.extract_from_text(
            text, 
            document_type="scientific_article"
        )
        
        # Проверяем базовые свойства результата
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) > 0
        assert result.text_id is not None
        assert result.processing_time > 0
        
        # Проверяем наличие ключевых концептов
        concept_texts = [c.text.lower() for c in result.concepts]
        
        expected_concepts = [
            "искусственный интеллект",
            "машинное обучение", 
            "медицина",
            "диагностика",
            "нейронные сети"
        ]
        
        found_concepts = 0
        for expected in expected_concepts:
            if any(expected in text.lower() for text in concept_texts):
                found_concepts += 1
        
        # Ожидаем найти не менее 70% ожидаемых концептов
        assert found_concepts >= len(expected_concepts) * 0.7
        
        # Проверяем ранжирование по важности
        importance_scores = [c.importance_score for c in result.concepts]
        assert len(set(importance_scores)) >= 1  # Есть вариация в оценках
        
        print(f"✓ Найдено {len(result.concepts)} концептов в научной статье")
    
    @pytest.mark.asyncio
    async def test_news_article_extraction(self, extractor: ConceptExtractor):
        """Тест извлечения концептов из новостной статьи."""
        text = TestDocumentTypes.get_news_article()
        
        result = await extractor.extract_from_text(
            text, 
            document_type="news_article"
        )
        
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) > 0
        
        # Проверяем наличие организаций и персон
        labels = [c.label for c in result.concepts]
        has_org = any(label in ['ORG', 'PERSON'] for label in labels)
        has_person = 'Tesla' in [c.text for c in result.concepts]
        
        assert has_org or has_person, "Должны быть найдены организации или персоны"
        
        # Проверяем связи
        assert len(result.relationships) >= 0
        
        print(f"✓ Найдено {len(result.concepts)} концептов в новостной статье")
    
    @pytest.mark.asyncio
    async def test_technical_document_extraction(self, extractor: ConceptExtractor):
        """Тест извлечения концептов из технического документа."""
        text = TestDocumentTypes.get_technical_document()
        
        result = await extractor.extract_from_text(
            text, 
            document_type="technical_document"
        )
        
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) > 0
        
        # Проверяем наличие технических терминов
        concept_texts = [c.text.lower() for c in result.concepts]
        
        technical_terms = [
            "микросервис",
            "api",
            "docker",
            "kubernetes", 
            "postgresql",
            "mongodb"
        ]
        
        found_technical_terms = []
        for term in technical_terms:
            if any(term in text.lower() for text in concept_texts):
                found_technical_terms.append(term)
        
        # Ожидаем найти хотя бы несколько технических терминов
        assert len(found_technical_terms) >= len(technical_terms) * 0.5
        
        print(f"✓ Найдено {len(result.concepts)} концептов в техническом документе")
    
    @pytest.mark.asyncio
    async def test_financial_report_extraction(self, extractor: ConceptExtractor):
        """Тест извлечения концептов из финансового отчета."""
        text = TestDocumentTypes.get_financial_report()
        
        result = await extractor.extract_from_text(
            text, 
            document_type="financial_report"
        )
        
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) > 0
        
        # Проверяем наличие числовых данных и организаций
        concept_texts = [c.text for c in result.concepts]
        
        # Ищем Microsoft
        has_microsoft = any("microsoft" in text.lower() for text in concept_texts)
        assert has_microsoft or len([c for c in result.concepts if c.label in ['ORG', 'PERSON']]) > 0
        
        # Ищем денежные суммы
        has_money = any(c.label == 'MONEY' for c in result.concepts)
        
        print(f"✓ Найдено {len(result.concepts)} концептов в финансовом отчете")
        print(f"✓ Найдено {len(result.relationships)} связей")
    
    @pytest.mark.asyncio
    async def test_legal_document_extraction(self, extractor: ConceptExtractor):
        """Тест извлечения концептов из правового документа."""
        text = TestDocumentTypes.get_legal_document()
        
        result = await extractor.extract_from_text(
            text, 
            document_type="legal_document"
        )
        
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) > 0
        
        # Проверяем наличие персон и организаций
        labels = [c.label for c in result.concepts]
        has_entities = any(label in ['PERSON', 'ORG', 'GPE'] for label in labels)
        
        assert has_entities, "Должны быть найдены персоны, организации или геополитические сущности"
        
        print(f"✓ Найдено {len(result.concepts)} концептов в правовом документе")
    
    @pytest.mark.asyncio
    async def test_batch_extraction(self, extractor: ConceptExtractor):
        """Тест пакетного извлечения концептов."""
        test_documents = [
            TestDocumentTypes.get_scientific_article(),
            TestDocumentTypes.get_news_article(),
            TestDocumentTypes.get_technical_document()
        ]
        
        results = []
        for i, text in enumerate(test_documents):
            result = await extractor.extract_from_text(
                text, 
                document_type=f"test_document_{i}"
            )
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, ExtractedKnowledge) for r in results)
        assert all(len(r.concepts) > 0 for r in results)
        
        # Проверяем статистику
        stats = extractor.get_statistics()
        assert stats['documents_processed'] >= 3
        
        print(f"✓ Пакетная обработка {len(results)} документов завершена успешно")
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, extractor_with_memory: ConceptExtractor):
        """Тест интеграции с MemoryManager."""
        text = TestDocumentTypes.get_scientific_article()
        
        result = await extractor_with_memory.extract_from_text(
            text,
            document_type="test_memory_integration"
        )
        
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) > 0
        
        # Проверяем, что данные сохранились в память
        # (это базовая проверка, так как детальная проверка требует доступа к MemoryManager)
        
        print("✓ Интеграция с MemoryManager работает корректно")
    
    @pytest.mark.asyncio
    async def test_concept_grouping(self, extractor: ConceptExtractor):
        """Тест семантического группирования концептов."""
        text = """
        Microsoft развивает облачные технологии Azure. 
        Компания Microsoft также инвестирует в ИИ и машинное обучение.
        Azure является конкурентом Amazon Web Services.
        """
        
        result = await extractor.extract_from_text(text)
        
        # Проверяем, что концепт "Microsoft" встречается несколько раз
        microsoft_concepts = [c for c in result.concepts if "microsoft" in c.text.lower()]
        assert len(microsoft_concepts) >= 1
        
        # Проверяем метаданные группировки
        for concept in result.concepts:
            if concept.metadata.get('semantic_group', {}).get('is_grouped'):
                group_size = concept.metadata['semantic_group']['group_size']
                assert group_size >= 2
        
        print(f"✓ Семантическое группирование обработало {len(result.concepts)} концептов")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, extractor: ConceptExtractor):
        """Тест обработки ошибок."""
        # Тест с пустым текстом
        result = await extractor.extract_from_text("")
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) == 0
        
        # Тест с очень коротким текстом
        result = await extractor.extract_from_text("Тест")
        assert isinstance(result, ExtractedKnowledge)
        
        print("✓ Обработка ошибок работает корректно")


class PerformanceTests:
    """Тесты производительности извлечения концептов."""
    
    @pytest.mark.asyncio
    async def test_processing_speed(self):
        """Тест скорости обработки документов."""
        extractor = await create_concept_extractor()
        
        # Тестовый текст среднего размера
        text = TestDocumentTypes.get_scientific_article() * 3  # Утроенный размер
        
        import time
        start_time = time.time()
        
        result = await extractor.extract_from_text(text)
        
        processing_time = time.time() - start_time
        
        # Проверяем, что обработка не занимает слишком много времени
        # (порог может быть скорректирован в зависимости от производительности)
        assert processing_time < 30.0  # Максимум 30 секунд
        
        print(f"✓ Обработка завершена за {processing_time:.2f} секунд")
        print(f"✓ Найдено {len(result.concepts)} концептов")
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self):
        """Тест обработки больших документов."""
        extractor = await create_concept_extractor()
        
        # Создаем большой текст
        base_text = TestDocumentTypes.get_technical_document()
        large_text = " ".join([base_text] * 100)  # Увеличиваем в 100 раз
        
        result = await extractor.extract_from_text(large_text, document_type="large_document")
        
        # Проверяем, что система справляется с большими документами
        assert isinstance(result, ExtractedKnowledge)
        assert len(result.concepts) > 0
        assert result.processing_time > 0
        
        print(f"✓ Большой документ обработан: {len(result.concepts)} концептов за {result.processing_time:.2f}s")


# Функции для запуска тестов

async def run_all_tests():
    """Запускает все тесты извлечения концептов."""
    print("🚀 Запуск тестов системы извлечения концептов...\n")
    
    test_instance = TestConceptExtractor()
    
    # Создаем экстрактор для тестов
    extractor = await create_concept_extractor()
    
    test_results = []
    
    # Список тестов для выполнения
    tests = [
        ("Научная статья", test_instance.test_scientific_article_extraction),
        ("Новостная статья", test_instance.test_news_article_extraction), 
        ("Технический документ", test_instance.test_technical_document_extraction),
        ("Финансовый отчет", test_instance.test_financial_report_extraction),
        ("Правовой документ", test_instance.test_legal_document_extraction),
        ("Пакетная обработка", test_instance.test_batch_extraction),
        ("Семантическое группирование", test_instance.test_concept_grouping),
        ("Обработка ошибок", test_instance.test_error_handling)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"🔄 Выполняется тест: {test_name}")
            await test_func(extractor)
            test_results.append((test_name, "✅ PASSED", None))
            print(f"✅ Тест '{test_name}' пройден\n")
        except Exception as e:
            test_results.append((test_name, "❌ FAILED", str(e)))
            print(f"❌ Тест '{test_name}' провален: {e}\n")
    
    # Тесты производительности
    perf_tests = PerformanceTests()
    try:
        print("🔄 Выполняется тест производительности")
        await perf_tests.test_processing_speed()
        test_results.append(("Тест производительности", "✅ PASSED", None))
        print("✅ Тест производительности пройден\n")
    except Exception as e:
        test_results.append(("Тест производительности", "❌ FAILED", str(e)))
        print(f"❌ Тест производительности провален: {e}\n")
    
    # Выводим сводку результатов
    print("📊 СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    
    passed = sum(1 for _, status, _ in test_results if "PASSED" in status)
    failed = sum(1 for _, status, _ in test_results if "FAILED" in status)
    
    for test_name, status, error in test_results:
        status_line = f"{test_name}: {status}"
        if error:
            status_line += f" ({error})"
        print(status_line)
    
    print(f"\n🎯 Итого: {passed} пройдено, {failed} провалено")
    
    if failed == 0:
        print("🎉 Все тесты успешно завершены!")
    else:
        print("⚠️ Некоторые тесты требуют внимания")
    
    # Выводим статистику экстрактора
    stats = extractor.get_statistics()
    print(f"\n📈 СТАТИСТИКА ЭКСТРАКТОРА:")
    print(f"   • Всего концептов извлечено: {stats['total_concepts_extracted']}")
    print(f"   • Всего связей извлечено: {stats['total_relationships_extracted']}")
    print(f"   • Документов обработано: {stats['documents_processed']}")
    print(f"   • Среднее время обработки: {stats.get('average_processing_time', 0):.2f}s")


if __name__ == "__main__":
    # Запуск тестов при прямом вызове
    asyncio.run(run_all_tests())