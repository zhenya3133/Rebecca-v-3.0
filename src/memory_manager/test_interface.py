"""
Модульные тесты для интерфейса MemoryManager.
Проверяют корректность работы всех основных функций.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch

from memory_manager_interface import (
    MemoryManager,
    MemoryLayer,
    MemoryFilter,
    MemoryItem,
    PerformanceOptimizer,
    LayerFactory,
    IMemoryManager
)


class TestPerformanceOptimizer(unittest.TestCase):
    """Тесты для оптимизатора производительности."""
    
    def setUp(self):
        self.optimizer = PerformanceOptimizer(max_cache_size=100, default_ttl=1.0)
    
    def test_cache_operations(self):
        """Тест основных операций кэша."""
        # Тест сохранения и получения
        self.optimizer.set("test_key", "test_value")
        result = self.optimizer.get("test_key")
        self.assertEqual(result, "test_value")
        
        # Тест TTL
        import time
        time.sleep(1.1)
        result = self.optimizer.get("test_key")
        self.assertIsNone(result)
    
    def test_cache_limits(self):
        """Тест ограничений кэша."""
        # Заполнение кэша сверх лимита
        for i in range(150):
            self.optimizer.set(f"key_{i}", f"value_{i}")
        
        # Проверка, что старые записи удалены
        old_keys = [f"key_{i}" for i in range(50)]
        for key in old_keys:
            self.assertIsNone(self.optimizer.get(key))
        
        # Новые записи должны быть доступны
        self.assertIsNotNone(self.optimizer.get("key_100"))
    
    def test_stats(self):
        """Тест статистики кэша."""
        # Добавление записей
        self.optimizer.set("hit", "value")
        self.optimizer.get("hit")  # Попадание
        self.optimizer.get("miss") # Промах
        
        stats = self.optimizer.get_stats()
        self.assertEqual(stats["hit_count"], 1)
        self.assertEqual(stats["miss_count"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)


class TestLayerFactory(unittest.TestCase):
    """Тесты для фабрики слоев памяти."""
    
    def setUp(self):
        # Сброс зарегистрированных слоев
        LayerFactory._layer_classes.clear()
    
    def test_registration_and_creation(self):
        """Тест регистрации и создания слоев."""
        class TestLayer:
            def __init__(self):
                self.name = "test_layer"
        
        # Регистрация
        LayerFactory.register_layer(MemoryLayer.CORE, TestLayer)
        
        # Создание
        layer = LayerFactory.create_layer(MemoryLayer.CORE)
        self.assertIsInstance(layer, TestLayer)
        self.assertEqual(layer.name, "test_layer")
    
    def test_unknown_layer_error(self):
        """Тест ошибки при создании неизвестного слоя."""
        with self.assertRaises(ValueError):
            LayerFactory.create_layer(MemoryLayer.CORE)


class TestMemoryItem(unittest.TestCase):
    """Тесты для структуры элемента памяти."""
    
    def test_creation(self):
        """Тест создания элемента памяти."""
        import time
        
        item = MemoryItem(
            id="test_id",
            layer=MemoryLayer.CORE,
            data={"key": "value"},
            metadata={"meta": "data"}
        )
        
        self.assertEqual(item.id, "test_id")
        self.assertEqual(item.layer, MemoryLayer.CORE)
        self.assertEqual(item.data, {"key": "value"})
        self.assertEqual(item.metadata, {"meta": "data"})
        self.assertIsInstance(item.timestamp, float)
        self.assertLessEqual(item.timestamp, time.time())


class TestMemoryFilter(unittest.TestCase):
    """Тесты для фильтров памяти."""
    
    def test_creation(self):
        """Тест создания фильтра."""
        filter_obj = MemoryFilter(
            metadata={"category": "test"},
            time_range=(1000.0, 2000.0)
        )
        
        self.assertEqual(filter_obj.metadata, {"category": "test"})
        self.assertEqual(filter_obj.time_range, (1000.0, 2000.0))
        self.assertIsNone(filter_obj.vector_similarity)


class TestIMemoryManager(unittest.TestCase):
    """Тесты для интерфейса IMemoryManager."""
    
    def test_interface_methods(self):
        """Тест наличия всех требуемых методов в интерфейсе."""
        required_methods = [
            'store', 'retrieve', 'update', 'delete', 'list_layers'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(IMemoryManager, method))
            self.assertTrue(callable(getattr(IMemoryManager, method)))


class TestMemoryManager(unittest.TestCase):
    """Тесты для реализации MemoryManager."""
    
    def setUp(self):
        self.memory_manager = MemoryManager()
    
    def test_initialization(self):
        """Тест инициализации менеджера памяти."""
        self.assertIsNotNone(self.memory_manager.context)
        self.assertIsNotNone(self.memory_manager.vector_store)
        self.assertIsNotNone(self.memory_manager.optimizer)
        self.assertGreater(len(self.memory_manager.layers), 0)
    
    def test_list_layers(self):
        """Тест получения списка слоев."""
        layers = self.memory_manager.list_layers()
        
        self.assertIsInstance(layers, list)
        self.assertGreater(len(layers), 0)
        self.assertIn(MemoryLayer.CORE, layers)
    
    def test_memory_stats(self):
        """Тест получения статистики памяти."""
        stats = self.memory_manager.get_memory_stats()
        
        required_keys = [
            'layers_count', 'available_layers', 'cache_stats',
            'indexed_items', 'metadata_keys'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['layers_count'], len(self.memory_manager.layers))
    
    @patch('asyncio.create_task')
    async def test_store_operation(self, mock_create_task):
        """Тест операции сохранения."""
        # Имитация корутины
        mock_create_task.return_value = asyncio.Future()
        mock_create_task.return_value.set_result("test_id")
        
        # Вызов метода
        result = await self.memory_manager.store(
            MemoryLayer.CORE,
            {"test": "data"},
            {"category": "test"}
        )
        
        self.assertIsInstance(result, str)
    
    @patch('asyncio.create_task')
    async def test_retrieve_operation(self, mock_create_task):
        """Тест операции извлечения."""
        # Имитация корутины
        mock_create_task.return_value = asyncio.Future()
        mock_create_task.return_value.set_result([])
        
        # Вызов метода
        result = await self.memory_manager.retrieve(
            MemoryLayer.CORE,
            "test query"
        )
        
        self.assertIsInstance(result, list)
    
    def test_cache_operations(self):
        """Тест операций с кэшем."""
        # Очистка кэша
        self.memory_manager.clear_cache()
        
        # Проверка статистики после очистки
        stats = self.memory_manager.optimizer.get_stats()
        self.assertEqual(stats['cache_size'], 0)
    
    def test_clear_all_data(self):
        """Тест очистки всех данных."""
        # Очистка всех данных не должна вызывать ошибок
        self.memory_manager.clear_all_data()
        
        # Проверка, что слои все еще доступны
        layers = self.memory_manager.list_layers()
        self.assertGreater(len(layers), 0)


class TestMemoryManagerIntegration(unittest.TestCase):
    """Интеграционные тесты для MemoryManager."""
    
    def setUp(self):
        self.memory_manager = MemoryManager()
    
    async def test_full_workflow(self):
        """Тест полного рабочего процесса."""
        # Сохранение данных
        item_id = await self.memory_manager.store(
            MemoryLayer.CORE,
            {"fact": "test fact"},
            {"category": "integration"}
        )
        
        # Поиск данных
        results = await self.memory_manager.retrieve(
            MemoryLayer.CORE,
            "test",
            MemoryFilter(metadata={"category": "integration"})
        )
        
        self.assertGreater(len(results), 0)
        
        # Обновление данных
        success = await self.memory_manager.update(
            MemoryLayer.CORE,
            item_id,
            {"fact": "updated fact"}
        )
        self.assertTrue(success)
        
        # Удаление данных
        success = await self.memory_manager.delete(
            MemoryLayer.CORE,
            item_id
        )
        self.assertTrue(success)


def run_async_test(test_func):
    """Запуск асинхронного теста."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_func())
    finally:
        loop.close()


def run_all_tests():
    """Запуск всех тестов."""
    # Создание набора тестов
    test_suite = unittest.TestSuite()
    
    # Добавление синхронных тестов
    test_classes = [
        TestPerformanceOptimizer,
        TestLayerFactory,
        TestMemoryItem,
        TestMemoryFilter,
        TestIMemoryManager,
        TestMemoryManager
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Добавление интеграционного теста (асинхронного)
    integration_test = TestMemoryManagerIntegration('test_full_workflow')
    test_suite.addTest(integration_test)
    
    # Запуск тестов
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Запуск тестов для MemoryManager интерфейса\n")
    print("=" * 60)
    
    success = run_all_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Все тесты прошли успешно!")
    else:
        print("❌ Некоторые тесты не прошли!")
    
    exit(0 if success else 1)