# Libs
from pathlib import Path
from importlib.resources import files

# Modules
from . import models, data

__all__ = [
    'models',
    'data',
    'baseline'
]

def baseline(task_type):
    """
    Копирует notebook шаблон в текущую рабочую директорию.
    
    Args:
        task_type (str): Тип задачи. Доступные варианты:
            - 'image_classification'
            - 'image_semantic_segmentation'
            - 'text_classification'
    
    Returns:
        Path: Путь к скопированному файлу
    
    Raises:
        ValueError: Если task_type не поддерживается
        FileNotFoundError: Если файл шаблона не найден
    """
    available_tasks = [
        'image_classification',
        'image_semantic_segmentation',
        'text_classification'
    ]
    
    if task_type not in available_tasks:
        raise ValueError(
            f"Неизвестный тип задачи '{task_type}'. "
            f"Доступные варианты: {', '.join(available_tasks)}"
        )
    
    # Получаем путь к файлу шаблона в пакете
    try:
        package_path = files('sefixlines')
        notebooks_path = package_path / 'notebooks'
        source_file = notebooks_path / f'{task_type}.ipynb'
        
        # Читаем содержимое файла из пакета
        source_content = source_file.read_bytes()
    except Exception as e:
        raise FileNotFoundError(
            f"Не удалось найти файл шаблона '{task_type}.ipynb' в пакете. "
            f"Ошибка: {e}"
        )
    
    # Определяем путь назначения в текущей рабочей директории
    destination = Path.cwd() / 'sefixline.ipynb'
    
    # Записываем файл
    destination.write_bytes(source_content)
    return destination