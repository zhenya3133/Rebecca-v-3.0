# -*- coding: utf-8 -*-
"""
Конфигурационный модуль для платформы Rebecca
"""

# Поддерживаемые форматы конфигурационных файлов
SUPPORTED_CONFIG_FORMATS = ['yaml', 'yml', 'json']

# Основные компоненты конфигурации
CONFIG_COMPONENTS = {
    'core': 'Основные настройки системы',
    'rebecca': 'Конфигурация мета-агента Rebecca',
    'agents': 'Настройки специализированных агентов',
    'memory': 'Конфигурация системы памяти',
    'logging': 'Настройки логирования',
    'security': 'Параметры безопасности'
}

# Импорты конфигурационных модулей
try:
    from .rebecca import RebeccaConfig, load_rebecca_config, validate_rebecca_config
    REBECCA_AVAILABLE = True
except ImportError as e:
    REBECCA_AVAILABLE = False
    REBECCA_IMPORT_ERROR = str(e)

# Основной класс конфигурации платформы
class PlatformConfig:
    """
    Центральный класс конфигурации для всей платформы Rebecca
    """
    
    def __init__(self, config_dir=None):
        """
        Инициализация конфигурации платформы
        
        Args:
            config_dir (str): Директория с конфигурационными файлами
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self._configurations = {}
        self._load_all_configurations()
    
    def _get_default_config_dir(self):
        """Получение директории по умолчанию для конфигурации"""
        import os
        return os.path.dirname(__file__)
    
    def _load_all_configurations(self):
        """Загрузка всех доступных конфигураций"""
        # Загрузка основной конфигурации системы
        try:
            from .core import CoreConfig
            self._configurations['core'] = CoreConfig(self.config_dir)
        except ImportError:
            pass
        
        # Загрузка конфигурации Rebecca
        if REBECCA_AVAILABLE:
            try:
                self._configurations['rebecca'] = RebeccaConfig(
                    self.config_dir + '/rebecca'
                )
            except Exception as e:
                print(f"Предупреждение: не удалось загрузить конфигурацию Rebecca: {e}")
        else:
            print(f"Предупреждение: модуль Rebecca недоступен: {REBECCA_IMPORT_ERROR}")
    
    def get_config(self, component_name):
        """
        Получение конфигурации компонента
        
        Args:
            component_name (str): Имя компонента
            
        Returns:
            Конфигурация компонента или None
        """
        return self._configurations.get(component_name)
    
    def get_rebecca_config(self):
        """
        Получение конфигурации мета-агента Rebecca
        
        Returns:
            RebeccaConfig или None
        """
        return self._configurations.get('rebecca')
    
    def get_core_config(self):
        """
        Получение основной конфигурации системы
        
        Returns:
            CoreConfig или None
        """
        return self._configurations.get('core')
    
    def list_available_components(self):
        """
        Получение списка доступных компонентов
        
        Returns:
            list: Список имен доступных компонентов
        """
        return list(self._configurations.keys())
    
    def validate_all_configurations(self):
        """
        Валидация всех загруженных конфигураций
        
        Returns:
            tuple: (is_valid, error_list)
        """
        all_valid = True
        all_errors = []
        
        for component_name, config in self._configurations.items():
            try:
                if hasattr(config, 'validate_configuration'):
                    is_valid, errors = config.validate_configuration()
                    if not is_valid:
                        all_valid = False
                        all_errors.extend([
                            f"{component_name}: {error}" for error in errors
                        ])
            except Exception as e:
                all_valid = False
                all_errors.append(f"{component_name}: Ошибка валидации - {str(e)}")
        
        return all_valid, all_errors
    
    def export_all_configurations(self, output_dir):
        """
        Экспорт всех конфигураций в директорию
        
        Args:
            output_dir (str): Директория для сохранения
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for component_name, config in self._configurations.items():
            try:
                if hasattr(config, 'export_configuration'):
                    output_path = os.path.join(output_dir, f"{component_name}_config.yaml")
                    config.export_configuration(output_path)
            except Exception as e:
                print(f"Ошибка экспорта конфигурации {component_name}: {e}")
    
    def reload_configurations(self):
        """Перезагрузка всех конфигураций"""
        self._configurations.clear()
        self._load_all_configurations()
    
    def get_component_info(self, component_name):
        """
        Получение информации о компоненте
        
        Args:
            component_name (str): Имя компонента
            
        Returns:
            dict: Информация о компоненте
        """
        if component_name not in CONFIG_COMPONENTS:
            return {"name": component_name, "description": "Неизвестный компонент"}
        
        config = self._configurations.get(component_name)
        
        info = {
            "name": component_name,
            "description": CONFIG_COMPONENTS[component_name],
            "available": config is not None
        }
        
        if config and hasattr(config, '__version__'):
            info["version"] = config.__version__
        
        if config and hasattr(config, 'get_available_agents'):
            info["agents"] = config.get_available_agents()
        
        if config and hasattr(config, 'get_available_workflows'):
            info["workflows"] = config.get_available_workflows()
        
        return info
    
    def list_all_components_info(self):
        """
        Получение информации о всех компонентах
        
        Returns:
            list: Список информации о компонентах
        """
        return [
            self.get_component_info(name) 
            for name in CONFIG_COMPONENTS.keys()
        ]
    
    def __str__(self):
        available_components = list(self._configurations.keys())
        return f"PlatformConfig(components={available_components})"
    
    def __repr__(self):
        return self.__str__()


# Функции-утилиты для работы с конфигурацией платформы
def load_platform_config(config_dir=None):
    """
    Удобная функция для загрузки конфигурации платформы
    
    Args:
        config_dir (str): Директория с конфигурационными файлами
        
    Returns:
        PlatformConfig: Объект конфигурации платформы
    """
    return PlatformConfig(config_dir)


def get_rebecca_config(config_dir=None):
    """
    Получение конфигурации мета-агента Rebecca
    
    Args:
        config_dir (str): Директория с конфигурационными файлами
        
    Returns:
        RebeccaConfig: Объект конфигурации Rebecca или None
    """
    platform_config = PlatformConfig(config_dir)
    return platform_config.get_rebecca_config()


def validate_platform_config(config_dir=None):
    """
    Валидация всей конфигурации платформы
    
    Args:
        config_dir (str): Путь к директории с конфигурацией
        
    Returns:
        tuple: (is_valid, error_list)
    """
    platform_config = PlatformConfig(config_dir)
    return platform_config.validate_all_configurations()


def list_platform_components(config_dir=None):
    """
    Получение списка доступных компонентов платформы
    
    Args:
        config_dir (str): Путь к директории с конфигурацией
        
    Returns:
        list: Список доступных компонентов
    """
    platform_config = PlatformConfig(config_dir)
    return platform_config.list_available_components()


def get_platform_info(config_dir=None):
    """
    Получение общей информации о платформе
    
    Args:
        config_dir (str): Путь к директории с конфигурацией
        
    Returns:
        dict: Информация о платформе
    """
    platform_config = PlatformConfig(config_dir)
    
    info = {
        "platform": "Rebecca Platform",
        "components": platform_config.list_all_components_info(),
        "available_components": platform_config.list_available_components()
    }
    
    # Добавляем информацию о Rebecca, если доступна
    rebecca_config = platform_config.get_rebecca_config()
    if rebecca_config:
        info["rebecca"] = {
            "available": True,
            "version": rebecca_config.agent_config.version if hasattr(rebecca_config, 'agent_config') else "unknown",
            "agents": rebecca_config.get_available_agents() if hasattr(rebecca_config, 'get_available_agents') else [],
            "workflows": rebecca_config.get_available_workflows() if hasattr(rebecca_config, 'get_available_workflows') else []
        }
    else:
        info["rebecca"] = {"available": False}
    
    return info


# Экспортируемые классы и функции
__all__ = [
    'PlatformConfig',
    'RebeccaConfig',
    'load_platform_config',
    'get_rebecca_config',
    'validate_platform_config',
    'list_platform_components',
    'get_platform_info',
    'SUPPORTED_CONFIG_FORMATS',
    'CONFIG_COMPONENTS',
    'REBECCA_AVAILABLE'
]