#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрационный пример использования конфигурации мета-агента Rebecca
"""

import sys
import os
import yaml
import json
from pathlib import Path

# Добавляем путь к конфигурации
config_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(config_root)

from rebecca import RebeccaConfig, load_rebecca_config, validate_rebecca_config

def demo_basic_usage():
    """Демонстрация базового использования конфигурации"""
    print("=" * 60)
    print("ДЕМО: Базовое использование конфигурации Rebecca")
    print("=" * 60)
    
    # Загрузка конфигурации по умолчанию
    print("\n1. Загрузка конфигурации по умолчанию...")
    config = load_rebecca_config()
    print(f"✓ Конфигурация загружена: {config}")
    
    # Проверка валидности
    print("\n2. Валидация конфигурации...")
    is_valid, errors = config.validate_configuration()
    if is_valid:
        print("✓ Конфигурация валидна")
    else:
        print("✗ Ошибки валидации:")
        for error in errors:
            print(f"  - {error}")
    
    # Получение доступных агентов
    print("\n3. Доступные агенты:")
    agents = config.get_available_agents()
    for agent in agents:
        print(f"  - {agent}")
    
    # Получение доступных workflow
    print("\n4. Доступные workflow:")
    workflows = config.get_available_workflows()
    for workflow in workflows:
        print(f"  - {workflow}")
    
    # Настройки таймаутов
    print("\n5. Настройки таймаутов:")
    timeouts = config.get_timeout_settings()
    for key, value in timeouts.items():
        print(f"  - {key}: {value} секунд")
    
    # Настройки повторных попыток
    print("\n6. Настройки повторных попыток:")
    retries = config.get_retry_settings()
    for key, value in retries.items():
        print(f"  - {key}: {value}")
    
    return config

def demo_agent_config(config):
    """Демонстрация работы с конфигурацией агентов"""
    print("\n" + "=" * 60)
    print("ДЕМО: Конфигурация специализированных агентов")
    print("=" * 60)
    
    # Конфигурация Backend агента
    print("\n1. Backend агент:")
    backend_config = config.get_agent_config("backend")
    print(f"  - Название: {backend_config.get('name', 'Не указано')}")
    print(f"  - Максимальное время: {backend_config.get('config', {}).get('max_execution_time', 'Не указано')} сек")
    print(f"  - Поддерживаемые языки: {backend_config.get('config', {}).get('allowed_languages', [])}")
    
    # Конфигурация Frontend агента
    print("\n2. Frontend агент:")
    frontend_config = config.get_agent_config("frontend")
    print(f"  - Название: {frontend_config.get('name', 'Не указано')}")
    print(f"  - Максимальное время: {frontend_config.get('config', {}).get('max_execution_time', 'Не указано')} сек")
    print(f"  - Поддерживаемые технологии: {frontend_config.get('config', {}).get('allowed_technologies', [])}")
    
    # Конфигурация ML агента
    print("\n3. ML агент:")
    ml_config = config.get_agent_config("ml")
    print(f"  - Название: {ml_config.get('name', 'Не указано')}")
    print(f"  - Максимальное время: {ml_config.get('config', {}).get('max_execution_time', 'Не указано')} сек")
    print(f"  - Поддержка GPU: {ml_config.get('config', {}).get('gpu_support', False)}")
    print(f"  - Поддерживаемые фреймворки: {ml_config.get('config', {}).get('allowed_frameworks', [])}")

def demo_prompts(config):
    """Демонстрация работы с промптами"""
    print("\n" + "=" * 60)
    print("ДЕМО: Шаблоны промптов")
    print("=" * 60)
    
    # Системный промпт
    print("\n1. Основной системный промпт:")
    main_prompt = config.get_prompt_template("system", "main")
    if main_prompt:
        # Показываем первые несколько строк
        lines = main_prompt.split('\n')[:3]
        for line in lines:
            print(f"  {line}")
        print("  ...")
    
    # Промпт планирования
    print("\n2. Промпт анализа задачи:")
    planning_prompt = config.get_prompt_template("planning", "task_analysis")
    if planning_prompt:
        lines = planning_prompt.split('\n')[:3]
        for line in lines:
            print(f"  {line}")
        print("  ...")
    
    # Инструкции для Backend агента
    print("\n3. Инструкции для Backend агента:")
    backend_instructions = config.get_prompt_template("agent_instruction", "backend")
    if backend_instructions:
        lines = backend_instructions.split('\n')[:3]
        for line in lines:
            print(f"  {line}")
        print("  ...")

def demo_workflows(config):
    """Демонстрация работы с workflow"""
    print("\n" + "=" * 60)
    print("ДЕМО: Шаблоны workflow")
    print("=" * 60)
    
    # Базовый workflow
    print("\n1. Базовый workflow:")
    basic_workflow = config.get_workflow_template("basic")
    if basic_workflow:
        print(f"  - Название: {basic_workflow.get('name', 'Не указано')}")
        print(f"  - Описание: {basic_workflow.get('description', 'Не указано')}")
        print(f"  - Количество шагов: {len(basic_workflow.get('steps', []))}")
        for i, step in enumerate(basic_workflow.get('steps', [])[:3]):
            print(f"    {i+1}. {step.get('name', 'Не указано')} (агент: {step.get('agent', 'Не указано')})")
        if len(basic_workflow.get('steps', [])) > 3:
            print(f"    ... и еще {len(basic_workflow.get('steps', [])) - 3} шагов")
    
    # Workflow для веб-разработки
    print("\n2. Workflow для веб-разработки:")
    web_workflow = config.get_workflow_template("web_development")
    if web_workflow:
        print(f"  - Название: {web_workflow.get('name', 'Не указано')}")
        print(f"  - Количество шагов: {len(web_workflow.get('steps', []))}")
    
    # ML workflow
    print("\n3. ML workflow:")
    ml_workflow = config.get_workflow_template("ml_project")
    if ml_workflow:
        print(f"  - Название: {ml_workflow.get('name', 'Не указано')}")
        print(f"  - Количество шагов: {len(ml_workflow.get('steps', []))}")

def demo_example_configs():
    """Демонстрация загрузки примеров конфигураций"""
    print("\n" + "=" * 60)
    print("ДЕМО: Примеры конфигураций")
    print("=" * 60)
    
    examples_dir = Path(__file__).parent / "examples"
    
    # Загрузка конфигурации для стартапа
    print("\n1. Конфигурация для стартапа:")
    startup_config_path = examples_dir / "startup_config.yaml"
    if startup_config_path.exists():
        with open(startup_config_path, 'r', encoding='utf-8') as f:
            startup_config = yaml.safe_load(f)
        meta_agent = startup_config.get('meta_agent', {})
        print(f"  - Название: {meta_agent.get('name', 'Не указано')}")
        print(f"  - Таймаут выполнения: {meta_agent.get('timeouts', {}).get('total_workflow', 'Не указано')} сек")
        print(f"  - Максимум агентов: {meta_agent.get('limits', {}).get('max_concurrent_agents', 'Не указано')}")
    
    # Загрузка конфигурации для предприятия
    print("\n2. Конфигурация для предприятия:")
    enterprise_config_path = examples_dir / "enterprise_config.yaml"
    if enterprise_config_path.exists():
        with open(enterprise_config_path, 'r', encoding='utf-8') as f:
            enterprise_config = yaml.safe_load(f)
        meta_agent = enterprise_config.get('meta_agent', {})
        print(f"  - Название: {meta_agent.get('name', 'Не указано')}")
        print(f"  - Таймаут выполнения: {meta_agent.get('timeouts', {}).get('total_workflow', 'Не указано')} сек")
        print(f"  - Максимум агентов: {meta_agent.get('limits', {}).get('max_concurrent_agents', 'Не указано')}")
        print(f"  - Память: {meta_agent.get('limits', {}).get('max_memory_usage_mb', 'Не указано')} MB")
    
    # Загрузка образовательной конфигурации
    print("\n3. Образовательная конфигурация:")
    education_config_path = examples_dir / "education_config.yaml"
    if education_config_path.exists():
        with open(education_config_path, 'r', encoding='utf-8') as f:
            education_config = yaml.safe_load(f)
        meta_agent = education_config.get('meta_agent', {})
        print(f"  - Название: {meta_agent.get('name', 'Не указано')}")
        print(f"  - Таймаут выполнения: {meta_agent.get('timeouts', {}).get('total_workflow', 'Не указано')} сек")
        integrations = education_config.get('integrations', {})
        educational_tools = integrations.get('educational_tools', [])
        print(f"  - Образовательные инструменты: {len(educational_tools)}")

def demo_export_import(config):
    """Демонстрация экспорта и импорта конфигурации"""
    print("\n" + "=" * 60)
    print("ДЕМО: Экспорт и импорт конфигурации")
    print("=" * 60)
    
    # Экспорт в JSON
    print("\n1. Экспорт конфигурации в JSON:")
    json_export_path = "/tmp/rebecca_config_demo.json"
    try:
        config.export_configuration(json_export_path)
        print(f"✓ Конфигурация экспортирована в {json_export_path}")
        
        # Проверяем размер файла
        file_size = os.path.getsize(json_export_path)
        print(f"  Размер файла: {file_size} байт")
    except Exception as e:
        print(f"✗ Ошибка экспорта: {e}")
    
    # Экспорт в YAML
    print("\n2. Экспорт конфигурации в YAML:")
    yaml_export_path = "/tmp/rebecca_config_demo.yaml"
    try:
        config.export_configuration(yaml_export_path)
        print(f"✓ Конфигурация экспортирована в {yaml_export_path}")
        
        # Проверяем размер файла
        file_size = os.path.getsize(yaml_export_path)
        print(f"  Размер файла: {file_size} байт")
    except Exception as e:
        print(f"✗ Ошибка экспорта: {e}")

def demo_platform_integration():
    """Демонстрация интеграции с платформой"""
    print("\n" + "=" * 60)
    print("ДЕМО: Интеграция с платформой")
    print("=" * 60)
    
    # Импортируем платформенную конфигурацию
    try:
        from config import PlatformConfig, get_platform_info
        
        print("\n1. Загрузка конфигурации платформы:")
        platform_config = PlatformConfig()
        print(f"✓ Платформа загружена: {platform_config}")
        
        print("\n2. Доступные компоненты:")
        components = platform_config.list_available_components()
        for component in components:
            print(f"  - {component}")
        
        print("\n3. Информация о платформе:")
        info = get_platform_info()
        print(f"  - Платформа: {info['platform']}")
        print(f"  - Доступные компоненты: {len(info['available_components'])}")
        if info.get('rebecca', {}).get('available'):
            rebecca_info = info['rebecca']
            print(f"  - Rebecca: доступна (версия {rebecca_info.get('version', 'unknown')})")
            print(f"  - Агенты: {len(rebecca_info.get('agents', []))}")
            print(f"  - Workflow: {len(rebecca_info.get('workflows', []))}")
        else:
            print(f"  - Rebecca: недоступна")
            
    except ImportError as e:
        print(f"✗ Ошибка импорта платформенной конфигурации: {e}")
    except Exception as e:
        print(f"✗ Ошибка загрузки платформы: {e}")

def main():
    """Основная функция демонстрации"""
    print("Демонстрация конфигурации мета-агента Rebecca")
    print("Версия: 2.0.0")
    print("Автор: Rebecca Platform Team")
    
    try:
        # Базовое использование
        config = demo_basic_usage()
        
        # Конфигурация агентов
        demo_agent_config(config)
        
        # Промпты
        demo_prompts(config)
        
        # Workflow
        demo_workflows(config)
        
        # Примеры конфигураций
        demo_example_configs()
        
        # Экспорт/импорт
        demo_export_import(config)
        
        # Интеграция с платформой
        demo_platform_integration()
        
        print("\n" + "=" * 60)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)