# -*- coding: utf-8 -*-
"""
Конфигурационный модуль для мета-агента Rebecca
"""

__version__ = "2.0.0"
__author__ = "Rebecca Platform Team"

import os
import yaml
import json
from pathlib import Path

# Мета-класс для конфигурации агента
class RebeccaConfig:
    """
    Центральный класс конфигурации для мета-агента Rebecca
    """
    
    def __init__(self, config_dir=None):
        """
        Инициализация конфигурации Rebecca
        
        Args:
            config_dir (str): Директория с конфигурационными файлами
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self._load_configurations()
    
    def _get_default_config_dir(self):
        """
        Получение директории по умолчанию для конфигурации
        
        Returns:
            str: Путь к директории конфигурации
        """
        return os.path.dirname(__file__)
    
    def _load_configurations(self):
        """Загрузка всех конфигурационных файлов"""
        # Загрузка основной конфигурации агента
        try:
            agent_config_path = os.path.join(self.config_dir, 'agent_config.yaml')
            with open(agent_config_path, 'r', encoding='utf-8') as f:
                self.agent_config_data = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки конфигурации агента: {e}")
        
        # Загрузка шаблонов промптов
        try:
            prompt_config_path = os.path.join(self.config_dir, 'prompt_templates.yaml')
            with open(prompt_config_path, 'r', encoding='utf-8') as f:
                self.prompt_templates_data = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки шаблонов промптов: {e}")
        
        # Загрузка шаблонов workflow
        try:
            workflow_config_path = os.path.join(self.config_dir, 'workflow_templates.yaml')
            with open(workflow_config_path, 'r', encoding='utf-8') as f:
                self.workflow_templates_data = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки шаблонов workflow: {e}")
    
    def get_agent_config(self, agent_type=None):
        """
        Получение конфигурации агента
        
        Args:
            agent_type (str): Тип агента (backend, frontend, ml, qa, devops)
            
        Returns:
            dict: Конфигурация агента
        """
        if agent_type:
            specialized_agents = self.agent_config_data.get('specialized_agents', {})
            return specialized_agents.get(agent_type, {})
        return self.agent_config_data.get('meta_agent', {})
    
    def get_prompt_template(self, template_type, template_name=None):
        """
        Получение шаблона промпта
        
        Args:
            template_type (str): Тип шаблона (system, planning, agent_instruction)
            template_name (str): Имя конкретного шаблона
            
        Returns:
            str или dict: Шаблон промпта
        """
        templates = self.prompt_templates_data.get(template_type, {})
        if template_name:
            template_data = templates.get(template_name, {})
            return template_data.get('template', '')
        return templates
    
    def get_workflow_template(self, workflow_type):
        """
        Получение шаблона workflow
        
        Args:
            workflow_type (str): Тип workflow (basic, web_development, ml_project)
            
        Returns:
            dict: Шаблон workflow
        """
        return self.workflow_templates_data.get('workflow_templates', {}).get(workflow_type, {})
    
    def validate_configuration(self):
        """
        Валидация всей конфигурации
        
        Returns:
            bool: True если конфигурация валидна
            list: Список ошибок валидации
        """
        errors = []
        
        # Базовая валидация - проверка наличия основных разделов
        required_sections = ['meta_agent', 'specialized_agents']
        if not hasattr(self, 'agent_config_data'):
            errors.append("Конфигурация агента не загружена")
        else:
            for section in required_sections:
                if section not in self.agent_config_data:
                    errors.append(f"Отсутствует раздел: {section}")
        
        if not hasattr(self, 'prompt_templates_data'):
            errors.append("Шаблоны промптов не загружены")
        
        if not hasattr(self, 'workflow_templates_data'):
            errors.append("Шаблоны workflow не загружены")
        
        return len(errors) == 0, errors
    
    def get_available_agents(self):
        """
        Получение списка доступных агентов
        
        Returns:
            list: Список типов агентов
        """
        specialized_agents = self.agent_config_data.get('specialized_agents', {})
        return list(specialized_agents.keys())
    
    def get_available_workflows(self):
        """
        Получение списка доступных workflow
        
        Returns:
            list: Список типов workflow
        """
        workflows = self.workflow_templates_data.get('workflow_templates', {})
        return list(workflows.keys())
    
    def get_timeout_settings(self):
        """
        Получение настроек таймаутов
        
        Returns:
            dict: Настройки таймаутов
        """
        return self.agent_config_data.get('meta_agent', {}).get('timeouts', {})
    
    def get_retry_settings(self):
        """
        Получение настроек повторных попыток
        
        Returns:
            dict: Настройки повторных попыток
        """
        return self.agent_config_data.get('meta_agent', {}).get('retries', {})
    
    def get_priority_rules(self):
        """
        Получение правил приоритизации задач
        
        Returns:
            list: Правила приоритизации
        """
        return self.agent_config_data.get('meta_agent', {}).get('planning', {}).get('priority_rules', [])
    
    def get_memory_settings(self):
        """
        Получение настроек памяти
        
        Returns:
            dict: Настройки памяти
        """
        return self.agent_config_data.get('memory', {})
    
    def get_communication_settings(self):
        """
        Получение настроек коммуникации
        
        Returns:
            dict: Настройки коммуникации
        """
        return self.agent_config_data.get('communication', {})
    
    def get_logging_settings(self):
        """
        Получение настроек логирования
        
        Returns:
            dict: Настройки логирования
        """
        return self.agent_config_data.get('logging', {})
    
    def get_security_settings(self):
        """
        Получение настроек безопасности
        
        Returns:
            dict: Настройки безопасности
        """
        return self.agent_config_data.get('security', {})
    
    def get_integration_settings(self, integration_type):
        """
        Получение настроек интеграции
        
        Args:
            integration_type (str): Тип интеграции
            
        Returns:
            dict: Настройки интеграции
        """
        return self.agent_config_data.get('integrations', {}).get(integration_type, {})
    
    def export_configuration(self, output_path):
        """
        Экспорт конфигурации в файл
        
        Args:
            output_path (str): Путь для сохранения конфигурации
        """
        config_export = {
            "agent_config": self.agent_config_data,
            "prompt_templates": self.prompt_templates_data,
            "workflow_templates": self.workflow_templates_data,
            "metadata": {
                "version": __version__,
                "exported_at": self._get_current_timestamp()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.endswith('.json'):
                json.dump(config_export, f, ensure_ascii=False, indent=2)
            elif output_path.endswith('.yaml') or output_path.endswith('.yml'):
                yaml.dump(config_export, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            else:
                raise ValueError("Поддерживаются только форматы JSON и YAML")
    
    def _get_current_timestamp(self):
        """Получение текущего timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def reload_configuration(self):
        """Перезагрузка конфигурации"""
        self._load_configurations()
    
    def __str__(self):
        return f"RebeccaConfig(version={__version__}, config_dir={self.config_dir})"
    
    def __repr__(self):
        return self.__str__()


# Функции-утилиты для работы с конфигурацией
def load_rebecca_config(config_dir=None):
    """
    Удобная функция для загрузки конфигурации Rebecca
    
    Args:
        config_dir (str): Директория с конфигурационными файлами
        
    Returns:
        RebeccaConfig: Объект конфигурации
    """
    return RebeccaConfig(config_dir)


def validate_rebecca_config(config_path=None):
    """
    Функция для валидации конфигурации Rebecca
    
    Args:
        config_path (str): Путь к директории с конфигурацией
        
    Returns:
        tuple: (is_valid, error_list)
    """
    config = load_rebecca_config(config_path)
    return config.validate_configuration()


def get_default_rebecca_config():
    """
    Получение стандартной конфигурации Rebecca
    
    Returns:
        RebeccaConfig: Объект конфигурации с настройками по умолчанию
    """
    return RebeccaConfig()


# Экспортируемые классы и функции
__all__ = [
    'RebeccaConfig',
    'load_rebecca_config',
    'validate_rebecca_config',
    'get_default_rebecca_config'
]