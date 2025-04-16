# 🚀 Модельный Пайплайн для Классификации

> "Хочешь быстро обучить пару моделей и не утонуть в кастомных велосипедах? Ты по адресу."

---

## 💡 Что это умеет?

- ⚡ Быстрый старт пайплайна для классификации изображений
- 🧠 Поддержка как `torchvision` моделей, так и `transformers` (ViT и иже с ними)
- 🛠️ Обёртка для `.fit()`, `.predict()`, `.save()`, `.load()` — без танцев
- 📈 LR Finder, визуализация предсказаний и простая адаптация под сабмит
- 🪄 Минимум кода, максимум гибкости (если знаешь, что делаешь)

---

## ⚙️ Установка

```bash
git clone https://github.com/Sefixnep/Torch-Classification.git
cd Torch-Classification
pip install -r requirements.txt
```

---

## 🔧 Что тебе нужно поменять

### 1. 📁 Данные

Задай пути к изображениям, классы и метки:

```python
classes = ["cat", "dog", "hedgehog"]  # свои классы
data = ["images/cat1.jpg", "images/dog2.jpg", ...]  # пути к изображениям
labels = [0, 1, ...]  # метки
```

### 2. 🎨 Аугментации (по желанию)

```python
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # можно добавить что угодно
])
```

### 3. 🧠 Выбор модели

Вариант 1: Transformers

```python
model = CustomOutput(
    AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(classes),
        ignore_mismatched_sizes=True
    )
)
```

Вариант 2: Torchvision models

```python
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
```

### 4. 🏁 Обучение

```python
model_wrapped = Classifier(model, "MyModel", optimizer)
model_wrapped.fit(train_loader, valid_loader, epochs=10)
```

### 5. 🧪 Предсказания

```python
test_dir = "path/to/test/images"
test_image_paths = [f"{test_dir}/{name}" for name in os.listdir(test_dir)]
test_set = Dataset(test_image_paths, transform)

pred_ids = best_model_wrapped.predict(test_set)
pred_names = [classes[i] for i in pred_ids]
```

---

## 🤝 Кто ты после этого?

Ты получаешь пайплайн, где можно:

- Заменить датасет — и сразу обучать
- Быстро подключить свои модели
- Не писать 300 строк под `.fit()`
- Устроить визуализацию, найти LR.

Если нужен bare-bones старт без фреймворков уровня "ещё один велосипед" — ты нашёл его.

---

> ❗ P.S. Всё кастомизируемо. Но по умолчанию уже работает.
