# 🚀 Модельные пайплайны **Sefixlines**

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
sefixlines.setup('image_classification')        # Классификация изображений
sefixlines.setup('image_semantic_segmentation') # Семантическая сегментация
sefixlines.setup('text_classification')         # Классификация текста
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
model_wrapper.fit(train_loader, valid_loader, num_epochs=3)

# Для семантической сегментации
segmenter = sefixlines.models.SemanticSegmenter(model, "MySegmenter")
segmenter.fit(train_loader, valid_loader, num_epochs=3)
```

## 🛠 Что можно настроить
- свой `optimizer`, `scheduler` или `loss_fn`
- аугментации в датасэте


---
> ❗ Есть идеи или нашли ошибку? Пишите в [telegram](https://t.me/sefixnep)
