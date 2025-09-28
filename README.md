# 🚀 Модельные пайплайны **Sefixlines**

> 🆕 **UPDATE**: классификация текста  
> 🆕 **UPDATE**: семантическая сегментация изображения

## ✨ Возможности
- ⚡ Быстрый старт без тонны кода
- 🖼️ Классификация изображений и текста
- 🎯 Семантическая сегментация изображения
- 💾 Автоматическое сохранение/загрузка весов
- 🔧 Простая кастомизация (loss_fn, optimizer, scheduler, augmentation)

## ⚙️ Установка
```bash
git clone https://github.com/Sefixnep/sefixlines.git
cd sefixlines
pip install -r requirements.txt
```

> ⚡ **Настоятельно рекомендуем ознакомиться с примерами решения вашей задачи в папке [notebooks](notebooks/)** — это поможет быстро разобраться и стартовать! 🌟

## 🚦 Минимальный запуск
1. **Подготовьте данные**
```python
# Классификация изображения
ImageClassificationDataset(paths, labels)

# Семантическая сегментация
ImageSemanticSegmentationDataset(image_paths, mask_paths)

# Классификация текста
TextClassificationDataset(texts, labels)
```
2. **Выберите модель** (любая модель, возвращающая логиты).
3. **Обучите**
```python
# Для классификации
model_wrapper = Classifier(model, "MyModel")
model_wrapper.fit(train_loader, valid_loader, num_epochs=10)

# Для семантической сегментации
segmenter = SemanticSegmenter(model, "MySegmenter")
segmenter.fit(train_loader, valid_loader, num_epochs=10)
```

## 🛠 Что можно настроить
- свой `optimizer`, `scheduler` или `loss_fn`
- аугментации в датасэте


---
> ❗ Есть идеи или нашли ошибку? Пишите в [telegram](https://t.me/sefixnep)
