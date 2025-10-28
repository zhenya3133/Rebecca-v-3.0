"""
Модуль GraphPersistence - обеспечивает сохранение и загрузку графа знаний.
Интегрируется с системой памяти Rebecca Platform и поддерживает различные форматы.
"""

from __future__ import annotations

import json
import pickle
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from .kag_graph import KAGGraph, Concept, Relationship, RelationshipType


class GraphPersistence:
    """
    Класс для управления персистентностью графа знаний.
    Обеспечивает сохранение и загрузку в различных форматах.
    """
    
    def __init__(self, graph: KAGGraph, storage_path: Optional[str] = None):
        """
        Инициализация системы персистентности.
        
        Args:
            graph: Экземпляр графа знаний
            storage_path: Путь для сохранения файлов
        """
        self.graph = graph
        self.storage_path = Path(storage_path) if storage_path else Path("data/kag_graph")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Метаданные
        self.metadata_file = self.storage_path / "metadata.json"
        self.backup_dir = self.storage_path / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Конфигурация
        self.config = {
            'auto_backup': True,
            'max_backups': 10,
            'compression': False,
            'incremental_save': True,
            'validation': True
        }
    
    def save_graph(
        self,
        format_type: str = "json",
        filename: Optional[str] = None,
        include_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Сохраняет граф в указанном формате.
        
        Args:
            format_type: Формат сохранения ("json", "pickle", "xml", "csv")
            filename: Имя файла (автоматически если не указано)
            include_metadata: Включать ли метаданные
            **kwargs: Дополнительные параметры
            
        Returns:
            Путь к сохраненному файлу
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"kag_graph_{timestamp}.{format_type}"
        
        filepath = self.storage_path / filename
        
        try:
            if format_type.lower() == "json":
                return self._save_json(filepath, include_metadata, **kwargs)
            elif format_type.lower() == "pickle":
                return self._save_pickle(filepath, include_metadata, **kwargs)
            elif format_type.lower() == "xml":
                return self._save_xml(filepath, include_metadata, **kwargs)
            elif format_type.lower() == "csv":
                return self._save_csv(filepath, include_metadata, **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format_type}")
        
        except Exception as e:
            print(f"Ошибка при сохранении графа: {e}")
            raise
    
    def load_graph(
        self,
        filename: str,
        format_type: Optional[str] = None,
        validate: bool = True,
        **kwargs
    ) -> bool:
        """
        Загружает граф из файла.
        
        Args:
            filename: Имя файла для загрузки
            format_type: Формат файла (автоматически определяется если не указан)
            validate: Валидировать ли данные при загрузке
            **kwargs: Дополнительные параметры
            
        Returns:
            True если успешно загружен
        """
        filepath = self.storage_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        # Определяем формат если не указан
        if not format_type:
            format_type = filepath.suffix[1:]  # Убираем точку
        
        try:
            if format_type.lower() == "json":
                return self._load_json(filepath, validate, **kwargs)
            elif format_type.lower() == "pickle":
                return self._load_pickle(filepath, validate, **kwargs)
            elif format_type.lower() == "xml":
                return self._load_xml(filepath, validate, **kwargs)
            elif format_type.lower() == "csv":
                return self._load_csv(filepath, validate, **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format_type}")
        
        except Exception as e:
            print(f"Ошибка при загрузке графа: {e}")
            return False
    
    def export_to_memory(self, memory_layer: str = "semantic") -> bool:
        """
        Экспортирует граф в систему памяти.
        
        Args:
            memory_layer: Слой памяти для экспорта
            
        Returns:
            True если успешно экспортирован
        """
        try:
            # Подготавливаем данные для экспорта
            export_data = self.graph.export_graph("json")
            
            # Экспортируем концепты
            for concept_data in export_data['concepts']:
                # Сохраняем в соответствующий слой памяти
                # (реализация зависит от конкретной системы памяти)
                print(f"Экспорт концепта: {concept_data['name']}")
            
            # Экспортируем отношения
            for relationship_data in export_data['relationships']:
                print(f"Экспорт отношения: {relationship_data['relationship_type']}")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при экспорте в память: {e}")
            return False
    
    def create_backup(self, description: str = "") -> str:
        """
        Создает резервную копию графа.
        
        Args:
            description: Описание резервной копии
            
        Returns:
            Имя созданного файла резервной копии
        """
        timestamp = int(time.time())
        backup_name = f"backup_{timestamp}_{description or 'auto'}.json"
        backup_path = self.backup_dir / backup_name
        
        try:
            # Создаем полную резервную копию
            export_data = self.graph.export_graph("json")
            export_data['backup_info'] = {
                'created_at': time.time(),
                'description': description,
                'original_graph_stats': self.graph.get_stats()
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            # Управляем количеством резервных копий
            self._manage_backups()
            
            return backup_name
            
        except Exception as e:
            print(f"Ошибка при создании резервной копии: {e}")
            raise
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """
        Восстанавливает граф из резервной копии.
        
        Args:
            backup_name: Имя файла резервной копии
            
        Returns:
            True если успешно восстановлен
        """
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Резервная копия не найдена: {backup_path}")
        
        try:
            return self.load_graph(backup_name, format_type="json")
            
        except Exception as e:
            print(f"Ошибка при восстановлении из резервной копии: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Возвращает список доступных резервных копий.
        
        Returns:
            Список информации о резервных копиях
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("backup_*.json"):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                backup_info = {
                    'filename': backup_file.name,
                    'filepath': str(backup_file),
                    'size': backup_file.stat().st_size,
                    'created_at': data.get('backup_info', {}).get('created_at', 0),
                    'description': data.get('backup_info', {}).get('description', ''),
                    'concepts_count': data.get('metadata', {}).get('total_concepts', 0),
                    'relationships_count': data.get('metadata', {}).get('total_relationships', 0)
                }
                
                backups.append(backup_info)
            
            except Exception as e:
                print(f"Ошибка при чтении информации о резервной копии {backup_file}: {e}")
                continue
        
        # Сортируем по дате создания (новые первыми)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        
        return backups
    
    def validate_graph_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидирует данные графа.
        
        Args:
            data: Данные для валидации
            
        Returns:
            Результат валидации
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Проверяем структуру данных
            if 'concepts' not in data:
                validation_result['errors'].append("Отсутствует секция 'concepts'")
                validation_result['is_valid'] = False
            
            if 'relationships' not in data:
                validation_result['errors'].append("Отсутствует секция 'relationships'")
                validation_result['is_valid'] = False
            
            if not validation_result['is_valid']:
                return validation_result
            
            concepts = data.get('concepts', [])
            relationships = data.get('relationships', [])
            
            # Валидируем концепты
            concept_ids = set()
            for i, concept_data in enumerate(concepts):
                if not isinstance(concept_data, dict):
                    validation_result['errors'].append(f"Концепт {i} не является словарем")
                    continue
                
                # Проверяем обязательные поля
                required_fields = ['id', 'name', 'description']
                for field in required_fields:
                    if field not in concept_data:
                        validation_result['errors'].append(
                            f"Концепт {i} отсутствует обязательное поле '{field}'"
                        )
                
                # Проверяем уникальность ID
                if 'id' in concept_data:
                    if concept_data['id'] in concept_ids:
                        validation_result['errors'].append(
                            f"Дублирование ID концепта: {concept_data['id']}"
                        )
                    concept_ids.add(concept_data['id'])
            
            # Валидируем отношения
            relationship_ids = set()
            for i, relationship_data in enumerate(relationships):
                if not isinstance(relationship_data, dict):
                    validation_result['errors'].append(f"Отношение {i} не является словарем")
                    continue
                
                # Проверяем обязательные поля
                required_fields = ['id', 'source_id', 'target_id', 'relationship_type']
                for field in required_fields:
                    if field not in relationship_data:
                        validation_result['errors'].append(
                            f"Отношение {i} отсутствует обязательное поле '{field}'"
                        )
                
                # Проверяем существование связанных концептов
                if 'source_id' in relationship_data and 'target_id' in relationship_data:
                    if relationship_data['source_id'] not in concept_ids:
                        validation_result['errors'].append(
                            f"Отношение {i}: исходный концепт {relationship_data['source_id']} не найден"
                        )
                    
                    if relationship_data['target_id'] not in concept_ids:
                        validation_result['errors'].append(
                            f"Отношение {i}: целевой концепт {relationship_data['target_id']} не найден"
                        )
                
                # Проверяем уникальность ID отношений
                if 'id' in relationship_data:
                    if relationship_data['id'] in relationship_ids:
                        validation_result['errors'].append(
                            f"Дублирование ID отношения: {relationship_data['id']}"
                        )
                    relationship_ids.add(relationship_data['id'])
            
            # Собираем статистику
            validation_result['stats'] = {
                'total_concepts': len(concepts),
                'total_relationships': len(relationships),
                'unique_concept_ids': len(concept_ids),
                'unique_relationship_ids': len(relationship_ids),
                'has_metadata': 'metadata' in data,
                'backup_info': 'backup_info' in data
            }
            
            # Генерируем предупреждения
            if len(concepts) == 0:
                validation_result['warnings'].append("Граф не содержит концептов")
            
            if len(relationships) == 0:
                validation_result['warnings'].append("Граф не содержит отношений")
            
            validation_result['is_valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f"Ошибка валидации: {str(e)}")
            validation_result['is_valid'] = False
        
        return validation_result
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику хранилища.
        
        Returns:
            Статистика файлов и резервных копий
        """
        stats = {
            'storage_path': str(self.storage_path),
            'main_files': [],
            'backups': [],
            'total_size': 0,
            'backup_count': 0
        }
        
        # Анализируем основные файлы
        for file_path in self.storage_path.glob("*.json"):
            if not file_path.name.startswith("backup_"):
                try:
                    file_stat = file_path.stat()
                    stats['main_files'].append({
                        'name': file_path.name,
                        'size': file_stat.st_size,
                        'modified': file_stat.st_mtime
                    })
                    stats['total_size'] += file_stat.st_size
                except Exception as e:
                    print(f"Ошибка при анализе файла {file_path}: {e}")
        
        # Анализируем резервные копии
        for backup_file in self.backup_dir.glob("*.json"):
            try:
                file_stat = backup_file.stat()
                stats['backups'].append({
                    'name': backup_file.name,
                    'size': file_stat.st_size,
                    'modified': file_stat.st_mtime
                })
                stats['total_size'] += file_stat.st_size
            except Exception as e:
                print(f"Ошибка при анализе резервной копии {backup_file}: {e}")
        
        stats['backup_count'] = len(stats['backups'])
        
        return stats
    
    # Приватные методы для работы с различными форматами
    
    def _save_json(self, filepath: Path, include_metadata: bool, **kwargs) -> str:
        """Сохраняет в формате JSON."""
        export_data = self.graph.export_graph("json")
        
        if include_metadata:
            export_data['export_info'] = {
                'exported_at': time.time(),
                'format': 'json',
                'graph_stats': self.graph.get_stats(),
                'file_hash': self._calculate_file_hash(export_data)
            }
        
        # Дополнительные опции JSON
        ensure_ascii = kwargs.get('ensure_ascii', False)
        indent = kwargs.get('indent', 2)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=ensure_ascii, indent=indent)
        
        return str(filepath)
    
    def _save_pickle(self, filepath: Path, include_metadata: bool, **kwargs) -> str:
        """Сохраняет в формате Pickle."""
        export_data = self.graph.export_graph("dict")
        
        if include_metadata:
            export_data['export_info'] = {
                'exported_at': time.time(),
                'format': 'pickle',
                'graph_stats': self.graph.get_stats()
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f, **kwargs)
        
        return str(filepath)
    
    def _save_xml(self, filepath: Path, include_metadata: bool, **kwargs) -> str:
        """Сохраняет в формате XML (базовая реализация)."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("Модуль xml.etree.ElementTree недоступен")
        
        # Создаем корневой элемент
        root = ET.Element("kag_graph")
        
        if include_metadata:
            export_info = ET.SubElement(root, "export_info")
            export_info.set("exported_at", str(time.time()))
            export_info.set("format", "xml")
            export_info.set("graph_stats", json.dumps(self.graph.get_stats()))
        
        # Добавляем концепты
        concepts_elem = ET.SubElement(root, "concepts")
        for concept in self.graph.concepts.values():
            concept_elem = ET.SubElement(concepts_elem, "concept")
            concept_elem.set("id", concept.id)
            concept_elem.set("name", concept.name)
            concept_elem.set("category", concept.category)
            
            description = ET.SubElement(concept_elem, "description")
            description.text = concept.description
            
            metadata = ET.SubElement(concept_elem, "metadata")
            metadata.text = json.dumps(concept.metadata)
        
        # Добавляем отношения
        relationships_elem = ET.SubElement(root, "relationships")
        for relationship in self.graph.relationships.values():
            rel_elem = ET.SubElement(relationships_elem, "relationship")
            rel_elem.set("id", relationship.id)
            rel_elem.set("source_id", relationship.source_id)
            rel_elem.set("target_id", relationship.target_id)
            rel_elem.set("relationship_type", relationship.relationship_type.value)
            rel_elem.set("strength", str(relationship.strength))
            
            description = ET.SubElement(rel_elem, "description")
            description.text = relationship.description
        
        # Сохраняем файл
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        return str(filepath)
    
    def _save_csv(self, filepath: Path, include_metadata: bool, **kwargs) -> str:
        """Сохраняет в формате CSV (концепты и отношения отдельно)."""
        import csv
        
        # Сохраняем концепты
        concepts_file = filepath.parent / f"{filepath.stem}_concepts.csv"
        with open(concepts_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'description', 'category', 'confidence_score', 'tags', 'metadata'])
            
            for concept in self.graph.concepts.values():
                writer.writerow([
                    concept.id,
                    concept.name,
                    concept.description,
                    concept.category,
                    concept.confidence_score,
                    '|'.join(concept.tags),
                    json.dumps(concept.metadata, ensure_ascii=False)
                ])
        
        # Сохраняем отношения
        relationships_file = filepath.parent / f"{filepath.stem}_relationships.csv"
        with open(relationships_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'source_id', 'target_id', 'relationship_type', 'strength', 'description', 'metadata'])
            
            for relationship in self.graph.relationships.values():
                writer.writerow([
                    relationship.id,
                    relationship.source_id,
                    relationship.target_id,
                    relationship.relationship_type.value,
                    relationship.strength,
                    relationship.description,
                    json.dumps(relationship.metadata, ensure_ascii=False)
                ])
        
        return str(filepath)
    
    def _load_json(self, filepath: Path, validate: bool, **kwargs) -> bool:
        """Загружает из формата JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if validate:
            validation_result = self.validate_graph_data(data)
            if not validation_result['is_valid']:
                print(f"Валидация не пройдена: {validation_result['errors']}")
                return False
        
        return self.graph.import_graph(data)
    
    def _load_pickle(self, filepath: Path, validate: bool, **kwargs) -> bool:
        """Загружает из формата Pickle."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if validate:
            validation_result = self.validate_graph_data(data)
            if not validation_result['is_valid']:
                print(f"Валидация не пройдена: {validation_result['errors']}")
                return False
        
        return self.graph.import_graph(data)
    
    def _load_xml(self, filepath: Path, validate: bool, **kwargs) -> bool:
        """Загружает из формата XML (базовая реализация)."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("Модуль xml.etree.ElementTree недоступен")
        
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Преобразуем XML в словарь
        data = {'concepts': [], 'relationships': []}
        
        # Загружаем концепты
        concepts_elem = root.find('concepts')
        if concepts_elem is not None:
            for concept_elem in concepts_elem.findall('concept'):
                concept_data = {
                    'id': concept_elem.get('id'),
                    'name': concept_elem.get('name'),
                    'description': concept_elem.find('description').text if concept_elem.find('description') is not None else '',
                    'category': concept_elem.get('category', ''),
                    'confidence_score': 1.0,
                    'tags': [],
                    'metadata': {}
                }
                
                metadata_elem = concept_elem.find('metadata')
                if metadata_elem is not None and metadata_elem.text:
                    try:
                        concept_data['metadata'] = json.loads(metadata_elem.text)
                    except json.JSONDecodeError:
                        pass
                
                data['concepts'].append(concept_data)
        
        # Загружаем отношения
        relationships_elem = root.find('relationships')
        if relationships_elem is not None:
            for rel_elem in relationships_elem.findall('relationship'):
                relationship_data = {
                    'id': rel_elem.get('id'),
                    'source_id': rel_elem.get('source_id'),
                    'target_id': rel_elem.get('target_id'),
                    'relationship_type': rel_elem.get('relationship_type'),
                    'strength': float(rel_elem.get('strength', 1.0)),
                    'description': rel_elem.find('description').text if rel_elem.find('description') is not None else '',
                    'metadata': {}
                }
                
                data['relationships'].append(relationship_data)
        
        if validate:
            validation_result = self.validate_graph_data(data)
            if not validation_result['is_valid']:
                print(f"Валидация не пройдена: {validation_result['errors']}")
                return False
        
        return self.graph.import_graph(data)
    
    def _load_csv(self, filepath: Path, validate: bool, **kwargs) -> bool:
        """Загружает из формата CSV."""
        import csv
        
        data = {'concepts': [], 'relationships': []}
        
        # Загружаем концепты
        concepts_file = filepath.parent / f"{filepath.stem}_concepts.csv"
        if concepts_file.exists():
            with open(concepts_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    concept_data = {
                        'id': row['id'],
                        'name': row['name'],
                        'description': row['description'],
                        'category': row['category'],
                        'confidence_score': float(row['confidence_score']),
                        'tags': row['tags'].split('|') if row['tags'] else [],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    }
                    data['concepts'].append(concept_data)
        
        # Загружаем отношения
        relationships_file = filepath.parent / f"{filepath.stem}_relationships.csv"
        if relationships_file.exists():
            with open(relationships_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    relationship_data = {
                        'id': row['id'],
                        'source_id': row['source_id'],
                        'target_id': row['target_id'],
                        'relationship_type': row['relationship_type'],
                        'strength': float(row['strength']),
                        'description': row['description'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    }
                    data['relationships'].append(relationship_data)
        
        if validate:
            validation_result = self.validate_graph_data(data)
            if not validation_result['is_valid']:
                print(f"Валидация не пройдена: {validation_result['errors']}")
                return False
        
        return self.graph.import_graph(data)
    
    def _manage_backups(self) -> None:
        """Управляет количеством резервных копий."""
        if not self.config['auto_backup']:
            return
        
        backup_files = list(self.backup_dir.glob("backup_*.json"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        max_backups = self.config['max_backups']
        if len(backup_files) > max_backups:
            for old_backup in backup_files[max_backups:]:
                try:
                    old_backup.unlink()
                    print(f"Удалена старая резервная копия: {old_backup.name}")
                except Exception as e:
                    print(f"Ошибка при удалении резервной копии {old_backup.name}: {e}")
    
    def _calculate_file_hash(self, data: Dict[str, Any]) -> str:
        """Вычисляет хеш файла для контроля целостности."""
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
