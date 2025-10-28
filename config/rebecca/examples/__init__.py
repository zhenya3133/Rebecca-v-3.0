# -*- coding: utf-8 -*-
"""
Примеры конфигурации для мета-агента Rebecca
"""

from pathlib import Path
import yaml
import json

# Доступные примеры конфигураций
AVAILABLE_EXAMPLES = {
    'startup': {
        'file': 'startup_config.yaml',
        'description': 'Конфигурация для стартапов',
        'use_case': 'Быстрая разработка с ограниченными ресурсами'
    },
    'enterprise': {
        'file': 'enterprise_config.yaml',
        'description': 'Конфигурация для крупных предприятий',
        'use_case': 'Высокие требования к качеству и масштабируемости'
    },
    'research': {
        'file': 'research_config.yaml',
        'description': 'Конфигурация для исследовательских проектов',
        'use_case': 'Долгие эксперименты и научные исследования'
    },
    'education': {
        'file': 'education_config.yaml',
        'description': 'Конфигурация для образовательных платформ',
        'use_case': 'Обучение и образовательные технологии'
    }
}

def load_example_config(example_name):
    """
    Загрузка примера конфигурации по имени
    
    Args:
        example_name (str): Имя примера (startup, enterprise, research, education)
        
    Returns:
        dict: Конфигурация примера
        
    Raises:
        ValueError: Если пример не найден
        FileNotFoundError: Если файл конфигурации не существует
    """
    if example_name not in AVAILABLE_EXAMPLES:
        raise ValueError(f"Пример '{example_name}' не найден. Доступные примеры: {list(AVAILABLE_EXAMPLES.keys())}")
    
    example_info = AVAILABLE_EXAMPLES[example_name]
    config_file = Path(__file__).parent / example_info['file']
    
    if not config_file.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def list_available_examples():
    """
    Получение списка доступных примеров конфигураций
    
    Returns:
        dict: Словарь с информацией о примерах
    """
    return AVAILABLE_EXAMPLES.copy()

def get_example_info(example_name):
    """
    Получение информации о конкретном примере
    
    Args:
        example_name (str): Имя примера
        
    Returns:
        dict: Информация о примере
        
    Raises:
        ValueError: Если пример не найден
    """
    if example_name not in AVAILABLE_EXAMPLES:
        raise ValueError(f"Пример '{example_name}' не найден. Доступные примеры: {list(AVAILABLE_EXAMPLES.keys())}")
    
    return AVAILABLE_EXAMPLES[example_name].copy()

def validate_example_config(example_name):
    """
    Валидация примера конфигурации
    
    Args:
        example_name (str): Имя примера
        
    Returns:
        tuple: (is_valid, error_list)
    """
    try:
        config = load_example_config(example_name)
        errors = []
        
        # Проверка основных разделов
        required_sections = ['meta_agent', 'specialized_agents']
        for section in required_sections:
            if section not in config:
                errors.append(f"Отсутствует обязательный раздел: {section}")
        
        # Проверка мета-агента
        if 'meta_agent' in config:
            meta_agent = config['meta_agent']
            if 'name' not in meta_agent:
                errors.append("В meta_agent отсутствует поле 'name'")
            if 'version' not in meta_agent:
                errors.append("В meta_agent отсутствует поле 'version'")
        
        # Проверка специализированных агентов
        if 'specialized_agents' in config:
            agents = config['specialized_agents']
            if not agents:
                errors.append("specialized_agents не должен быть пустым")
            else:
                for agent_name, agent_config in agents.items():
                    if 'name' not in agent_config:
                        errors.append(f"В агенте '{agent_name}' отсутствует поле 'name'")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"Ошибка загрузки конфигурации: {str(e)}"]

def export_example_config(example_name, output_path, format='yaml'):
    """
    Экспорт примера конфигурации в файл
    
    Args:
        example_name (str): Имя примера
        output_path (str): Путь для сохранения
        format (str): Формат файла ('yaml' или 'json')
        
    Raises:
        ValueError: Если пример не найден или формат не поддерживается
    """
    if example_name not in AVAILABLE_EXAMPLES:
        raise ValueError(f"Пример '{example_name}' не найден")
    
    if format not in ['yaml', 'json']:
        raise ValueError(f"Неподдерживаемый формат '{format}'. Поддерживаемые: yaml, json")
    
    config = load_example_config(example_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'json':
            json.dump(config, f, ensure_ascii=False, indent=2)
        elif format == 'yaml':
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

def create_custom_config_from_example(example_name, customizations=None):
    """
    Создание кастомной конфигурации на основе примера
    
    Args:
        example_name (str): Имя примера
        customizations (dict): Кастомизации для применения
        
    Returns:
        dict: Кастомная конфигурация
    """
    config = load_example_config(example_name)
    
    if customizations:
        # Глубокое слияние конфигураций
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        config = deep_merge(config, customizations)
    
    return config

def get_example_characteristics(example_name):
    """
    Получение характеристик примера конфигурации
    
    Args:
        example_name (str): Имя примера
        
    Returns:
        dict: Характеристики примера
    """
    if example_name not in AVAILABLE_EXAMPLES:
        raise ValueError(f"Пример '{example_name}' не найден")
    
    try:
        config = load_example_config(example_name)
        characteristics = {}
        
        # Характеристики мета-агента
        if 'meta_agent' in config:
            meta = config['meta_agent']
            characteristics.update({
                'name': meta.get('name', 'Неизвестно'),
                'version': meta.get('version', 'Неизвестно'),
                'total_timeout': meta.get('timeouts', {}).get('total_workflow', 'Не указано'),
                'max_agents': meta.get('limits', {}).get('max_concurrent_agents', 'Не указано'),
                'memory_limit': meta.get('limits', {}).get('max_memory_usage_mb', 'Не указано'),
                'max_tasks': meta.get('planning', {}).get('max_tasks', 'Не указано')
            })
        
        # Характеристики агентов
        if 'specialized_agents' in config:
            agents = config['specialized_agents']
            characteristics['agents'] = list(agents.keys())
            characteristics['agent_count'] = len(agents)
        
        # Характеристики памяти
        if 'memory' in config:
            memory = config['memory']
            characteristics['memory_layers'] = memory.get('layers_used', [])
            characteristics['cache_enabled'] = memory.get('cache_settings', {}).get('enabled', False)
            characteristics['max_cache_size'] = memory.get('cache_settings', {}).get('max_size_mb', 'Не указано')
        
        # Характеристики безопасности
        if 'security' in config:
            security = config['security']
            characteristics['auth_required'] = security.get('authentication', {}).get('required', False)
            characteristics['encryption_enabled'] = security.get('encryption', {}).get('enabled', False)
        
        # Характеристики интеграций
        if 'integrations' in config:
            integrations = config['integrations']
            characteristics['integrations'] = list(integrations.keys())
            characteristics['integration_count'] = len(integrations)
        
        return characteristics
        
    except Exception as e:
        return {'error': f"Ошибка анализа конфигурации: {str(e)}"}

# Функции для работы с демо
def run_demo():
    """
    Запуск демонстрации использования примеров конфигурации
    """
    try:
        from .demo_usage import main as demo_main
        return demo_main()
    except ImportError:
        print("Ошибка: не удалось импортировать демо-модуль")
        return 1

# Экспортируемые функции
__all__ = [
    'load_example_config',
    'list_available_examples',
    'get_example_info',
    'validate_example_config',
    'export_example_config',
    'create_custom_config_from_example',
    'get_example_characteristics',
    'run_demo',
    'AVAILABLE_EXAMPLES'
]