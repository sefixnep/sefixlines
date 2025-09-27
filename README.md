# 🚀 Модельные пайплайны **Sefixlines**

> 🆕 **UPDATE**: классификация текста  
> 🆕 **UPDATE**: семантическая сегментация изображения

## ✨ Возможности
- ⚡ Быстрый старт без тонны кода
- 🖼️ Классификация изображений и текста
- 🎯 Семантическая сегментация изображения
- 💾 Автоматическое сохранение/загрузка весов
- 🔧 Простая кастомизация (оптимизаторы, scheduler, аугментации)

## ⚙️ Установка
```bash
git clone https://github.com/Sefixnep/sefixlines.git
cd sefixlines
pip install -r requirements.txt
```

## 🚦 Минимальный запуск
1. **Подготовьте данные**
```python
# изображения
train_dataset = ImageClassificationDataset(train_paths, train_labels, augment=True)

# тексты
text_dataset = TextClassificationDataset(texts, labels)
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
- `ImageClassificationDataset.augmentation` — свои аугментации
- `ImageClassificationDataset.change_image_size((256, 256))`
- `TextClassificationDataset.tokenizer` и `max_length`
- свой `optimizer`, `scheduler` или `loss_fn` в `Classifier` и `SemanticSegmenter`
- `answer='masks'` для сегментации или `answer='labels'` для классификации
- поиграться с примерами в папке [notebooks](notebooks/) 🌟


---
> ❗ Есть идеи или нашли ошибку? Пишите в [telegram](https://t.me/sefixnep)
