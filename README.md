# 🚀 Модельные пайплайны 2.0

> 🆕 **UPDATE**: репозиторий обновлён — поддержка текста и ещё больше гибкости!

<p align="left">
  <a href="https://www.youtube.com/watch?v=pzMqIQdTnek" target="_blank">
    <img src="assets/screencast_preview.png" alt="Watch the video" style="width:100%; max-width:600px; border-radius:10px;">
  </a>
</p>

## ✨ Возможности
- ⚡ Быстрый старт без тонны кода
- 🖼️ Классификация изображений и ✍️ текста
- 📈 LR Finder и наглядные графики обучения
- 💾 Автоматическое сохранение/загрузка весов
- 🔧 Простая кастомизация (оптимизаторы, scheduler, аугментации)

## ⚙️ Установка
```bash
git clone https://github.com/Sefixnep/sefixnep_pipelines.git
cd sefixnep_pipelines
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
wrapper = Classifier(model, "MyModel")
wrapper.fit(train_loader, valid_loader, num_epochs=10)
```

## 🛠 Что можно настроить
- `ImageClassificationDataset.augmentation` — свои аугментации
- `ImageClassificationDataset.change_image_size((256, 256))`
- `TextClassificationDataset.tokenizer` и `max_length`
- свой `optimizer`, `scheduler` или `loss_fn` в `Classifier`
- поиграться с примерами в папке [notebooks](notebooks/) 🌟

---
> ❗ Есть идеи или нашли ошибку? Пишите в [telegram](https://t.me/sefixnep)
