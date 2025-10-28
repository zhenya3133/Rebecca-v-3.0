#!/usr/bin/env python3
"""
Демонстрационный пример системы извлечения концептов и связей для Rebecca-Platform.

Запустите: python demo_concept_extractor.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from knowledge_graph.concept_extractor import create_concept_extractor

async def demo_basic_extraction():
    """Базовая демонстрация извлечения концептов."""
    print("🚀 ДЕМОНСТРАЦИЯ СИСТЕМЫ ИЗВЛЕЧЕНИЯ КОНЦЕПТОВ")
    print("=" * 60)
    
    # Создаем экстрактор
    extractor = await create_concept_extractor()
    print("✅ Экстрактор концептов создан")
    
    # Тестовые тексты
    test_texts = [
        {
            "title": "🔬 Научная статья об ИИ",
            "text": """
            Искусственный интеллект революционизирует медицинскую диагностику.
            Компания Google DeepMind разработала систему AlphaFold для анализа белков.
            Исследователи из MIT используют нейронные сети для лечения рака.
            """
        },
        {
            "title": "📰 Новость о технологиях",
            "text": """
            Tesla под руководством Илона Маска анонсировала новый автопилот.
            Компания планирует запустить производство в 2024 году на заводе в Берлине.
            Акции Tesla выросли на 15% после объявления.
            """
        },
        {
            "title": "💼 Бизнес-отчет",
            "text": """
            Microsoft сообщила о росте выручки на 20% в Q3 2024.
            Облачные сервисы Azure принесли компании 32 миллиарда долларов.
            Генеральный директор Сатья Наделла назвал результаты выдающимися.
            """
        }
    ]
    
    for i, test_data in enumerate(test_texts, 1):
        print(f"\n📄 Текст {i}: {test_data['title']}")
        print("-" * 40)
        
        try:
            # Извлекаем концепты
            result = await extractor.extract_from_text(
                test_data['text'],
                document_type="demo"
            )
            
            print(f"⏱️ Время обработки: {result.processing_time:.2f}s")
            print(f"🎯 Найдено концептов: {len(result.concepts)}")
            print(f"🔗 Найдено связей: {len(result.relationships)}")
            
            if result.concepts:
                print("\n🏆 Топ концептов по важности:")
                # Сортируем по важности
                sorted_concepts = sorted(
                    result.concepts, 
                    key=lambda x: x.importance_score, 
                    reverse=True
                )
                
                for j, concept in enumerate(sorted_concepts[:5], 1):
                    print(f"  {j}. {concept.text} ({concept.label}) "
                          f"- важность: {concept.importance_score:.2f} "
                          f"- частота: {concept.frequency}")
            
            if result.relationships:
                print("\n🔗 Найденные связи:")
                for rel in result.relationships[:3]:
                    print(f"  • {rel.relationship_type} "
                          f"(уверенность: {rel.confidence:.2f})")
            
            print("✅ Обработка завершена успешно")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    # Статистика работы
    stats = extractor.get_statistics()
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА")
    print("=" * 30)
    print(f"Всего концептов извлечено: {stats['total_concepts_extracted']}")
    print(f"Всего связей извлечено: {stats['total_relationships_extracted']}")
    print(f"Документов обработано: {stats['documents_processed']}")
    if stats.get('average_processing_time'):
        print(f"Среднее время обработки: {stats['average_processing_time']:.2f}s")
    
    print("\n🎉 Демонстрация завершена!")

async def demo_advanced_features():
    """Демонстрация продвинутых возможностей."""
    print("\n\n🚀 ДЕМОНСТРАЦИЯ ПРОДВИНУТЫХ ВОЗМОЖНОСТЕЙ")
    print("=" * 60)
    
    extractor = await create_concept_extractor()
    
    # Сложный текст для демонстрации семантического группирования
    complex_text = """
    Искусственный интеллект меняет медицинскую диагностику. 
    AI системы анализируют рентгеновские снимки лучше врачей.
    Машинное обучение позволяет создавать персонализированные лекарства.
    Нейронные сети используются в онкологии для раннего выявления рака.
    Microsoft, Google и IBM инвестируют миллиарды в AI технологии.
    """
    
    print("📝 Обработка сложного текста с семантическим группированием...")
    
    try:
        result = await extractor.extract_from_text(
            complex_text,
            text_id="complex_demo",
            document_type="advanced_demo"
        )
        
        print(f"✅ Найдено {len(result.concepts)} концептов")
        
        # Анализ группировки
        grouped_concepts = 0
        for concept in result.concepts:
            if concept.metadata.get('semantic_group', {}).get('is_grouped'):
                grouped_concepts += 1
        
        if grouped_concepts > 0:
            print(f"🔗 Сгруппировано концептов: {grouped_concepts}")
            
            # Показываем группы
            for concept in result.concepts:
                group_info = concept.metadata.get('semantic_group', {})
                if group_info.get('is_grouped'):
                    print(f"  • Группа '{concept.text}': {group_info['group_size']} концептов")
        
        # Анализ типов сущностей
        entity_types = {}
        for concept in result.concepts:
            label = concept.label
            entity_types[label] = entity_types.get(label, 0) + 1
        
        print("\n📈 Типы найденных сущностей:")
        for entity_type, count in sorted(entity_types.items()):
            print(f"  • {entity_type}: {count}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

async def demo_error_handling():
    """Демонстрация обработки ошибок."""
    print("\n\n🛡️ ДЕМОНСТРАЦИЯ ОБРАБОТКИ ОШИБОК")
    print("=" * 50)
    
    extractor = await create_concept_extractor()
    
    test_cases = [
        ("Пустой текст", ""),
        ("Короткий текст", "Тест"),
        ("Специальные символы", "!@#$%^&*()"),
        ("Только числа", "123 456 789"),
        ("Очень длинный текст", "Слово " * 1000)
    ]
    
    for name, text in test_cases:
        print(f"\n🔍 Тест: {name}")
        try:
            result = await extractor.extract_from_text(text)
            print(f"  ✅ Обработано: {len(result.concepts)} концептов")
            if result.processing_time > 0:
                print(f"  ⏱️ Время: {result.processing_time:.3f}s")
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

async def main():
    """Главная функция демонстрации."""
    print("🤖 СИСТЕМА ИЗВЛЕЧЕНИЯ КОНЦЕПТОВ И СВЯЗЕЙ")
    print("📅 Rebecca-Platform - Демонстрация возможностей")
    print("=" * 60)
    
    try:
        await demo_basic_extraction()
        await demo_advanced_features()
        await demo_error_handling()
        
        print("\n\n🎯 ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("📚 Подробная документация: /workspace/reports/concept_extraction_implementation.md")
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Запуск демонстрации
    success = asyncio.run(main())
    sys.exit(0 if success else 1)