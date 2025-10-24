<div align="center">

# 🚀 Модельные пайплайны Sefixlines

<img src="https://raw.githubusercontent.com/sefixnep/sefixlines/main/assets/logo.png" alt="Sefixlines Logo" width="500"/>

<br>

[![PyPI](https://img.shields.io/badge/PyPI-Install%20Package-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/sefixlines/) &nbsp; [![GitHub](https://img.shields.io/badge/GitHub-View%20Source-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sefixnep/sefixlines)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/sefixnep) &nbsp; [![GitHub Profile](https://img.shields.io/badge/GitHub%20Profile-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sefixnep)

</div>

> 🆕 **UPDATE**: мультилейбл классификация  
> 🆕 **UPDATE**: классификация текста  

## ✨ Возможности
- ⚡ Быстрый старт без тонны кода
- 🖼️ Классификация изображений и текста
- 🎯 Семантическая сегментация изображения
- 💾 Автоматическое сохранение/загрузка весов
- 🔧 Простая кастомизация (loss_fn, optimizer, scheduler, augmentation)

## ⚙️ Установка
```bash
pip install sefixlines
```

## 🎯 Начни с базового примера

Для быстрого старта используйте готовые шаблоны с настроенными пайплайнами:

```python
import sefixlines

# Создаёт готовый notebook с примером для вашей задачи
sefixlines.baseline('image_classification')        # Классификация изображений
sefixlines.baseline('image_semantic_segmentation') # Семантическая сегментация
sefixlines.baseline('text_classification')         # Классификация текста
```

Эта команда создаст файл `sefixline.ipynb` в текущей директории с полностью рабочим примером, включая:
- 📊 Загрузку и подготовку данных
- 🤖 Настройку модели
- 🏋️ Обучение с визуализацией
- 📈 Оценку результатов

> ⚡ **Это самый быстрый способ начать работу!** Просто откройте созданный notebook и адаптируйте под свои данные.

## 🚦 Минимальный запуск вручную
1. **Подготовьте данные**
```python
# Классификация изображения
sefixlines.data.ImageClassificationDataset(paths, labels)

# Семантическая сегментация
sefixlines.data.ImageSemanticSegmentationDataset(image_paths, mask_paths)

# Классификация текста
sefixlines.data.TextClassificationDataset(texts, labels)
```
2. **Выберите модель** (любая модель, возвращающая логиты).
3. **Обучите**
```python
# Для классификации
model_wrapper = sefixlines.models.Classifier(model, "MyModel")
model_wrapper.fit(train_set, valid_set, num_epochs=3)

# Для семантической сегментации
segmenter = sefixlines.models.SemanticSegmenter(model, "MySegmenter")
segmenter.fit(train_set, valid_set, num_epochs=3)
```

## 🛠 Что можно настроить
- свой `optimizer`, `scheduler` или `loss_fn`
- аугментации в датасэте



Лицензия
--------

MIT. См. файл LICENSE.
