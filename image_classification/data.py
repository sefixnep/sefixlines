# Работа с данными
import numpy as np

# Torch
import torch
from torch.utils.data import Dataset

# Визуализация
import matplotlib.pyplot as plt

# Остальное
import os
import random
from PIL import Image

# Config
from config import *


def set_all_seeds(seed=42):
    # Устанавливаем seed для встроенного генератора Python
    random.seed(seed)
    # Устанавливаем seed для хэш-функции Python (опция для контроля поведения хэшей)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Устанавливаем seed для NumPy
    np.random.seed(seed)

    # Устанавливаем seed для PyTorch
    torch.manual_seed(seed)
    # Устанавливаем seed для генератора на CUDA
    torch.cuda.manual_seed(seed)
    # Отключаем недетерминированное поведение в алгоритмах CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(image_tensor):
    # Преобразуем mean и std в тензоры и переносим их на то же устройство, что и image
    tensor_mean = torch.tensor(mean).view(-1, 1, 1).to(image_tensor.device)
    tensor_std = torch.tensor(std).view(-1, 1, 1).to(image_tensor.device)

    # Денормализация: (тензор * std) + mean
    denormalize_image = image_tensor * tensor_std + tensor_mean

    # Преобразуем в диапазон [0, 255] и к типу uint8
    return (denormalize_image * 255).clamp(0, 255).byte()


def show_images(dataset, amount=3, figsize=(4, 4), classes=None, n_classes=None):
    # Получаем метки из dataset
    labels = np.array(dataset.labels)

    # Находим уникальные классы
    unique_classes = sorted(set(labels))[:n_classes]
    rows, cols = amount, len(unique_classes)

    # Изменяем figsize
    figsize = (figsize[0] * cols, figsize[1] * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_alpha(0.0)  # Прозрачный фон

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    shown_indices = {cls: 0 for cls in unique_classes}  # Отслеживаем, сколько картинок каждого класса показано

    for row in range(rows):
        for col, cls in enumerate(unique_classes):
            # Найдем индексы всех изображений текущего класса
            class_indices = np.where(labels == cls)[0]

            # Проверяем, сколько уже показано, и берем следующий индекс
            idx = class_indices[shown_indices[cls] % len(class_indices)]  # Циклично берем следующий индекс
            shown_indices[cls] += 1  # Увеличиваем счетчик для текущего класса

            # Загружаем изображение и метку
            image, label = dataset[idx]

            # Определяем название класса
            class_name = f"Class: {cls}" if classes is None else f"Class: {classes[cls]}"

            # Отображение изображения
            ax = axes[row][col]
            ax.imshow(denormalize(image).cpu().numpy().transpose(1, 2, 0))

            ax.set_title(class_name, fontsize=10, color='white')  # Белый текст для контраста
            ax.axis("off")

    # Отключаем лишние оси, если изображений меньше, чем ячеек
    for i in range(rows * cols, len(axes.flatten())):
        axes.flatten()[i].axis("off")

    plt.tight_layout()
    plt.show()


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Считываем изображение
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Применяем трансформации
        image = self.transform(image)

        return image.to(device)


class ImageClassificationDataset(ImageDataset):
    def __init__(self, image_paths, labels, transform):
        super().__init__(image_paths, transform)
        self.labels = labels

    def __getitem__(self, idx):
        # Получаем метку
        label = self.labels[idx]

        return super().__getitem__(idx), torch.tensor(label, dtype=torch.long).to(device)
