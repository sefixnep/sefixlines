# Работа с данными
import numpy as np

# Torch
import torch
from torch.utils.data import Dataset

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Остальное
import os
import random
from PIL import Image
from tqdm.notebook import tqdm

# Config
from notebooks.image_classification.config import *


def set_all_seeds(seed=42):
    """
    Устанавливает фиксированные значения для всех возможных генераторов случайных чисел.

    Parameters
    ----------
    seed : int, optional
        Значение, которое будет использоваться в качестве начального для генерации случайных чисел (по умолчанию 42).

    Notes
    -----
    - Устанавливает значения для генераторов случайных чисел в Python, NumPy и PyTorch.
    - Обеспечивает воспроизводимость экспериментов, включая случайное поведение на GPU.
    """
    # python's seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # torch's seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(image_tensor):
    """
    Преобразует нормализованный тензор изображения обратно в оригинальный диапазон.

    Parameters
    ----------
    image_tensor : torch.tensor
        Нормализованный тензор изображения (значения в диапазоне [0, 1]).

    Returns
    -------
    torch.tensor
        Денормализованный тензор изображения (значения в диапазоне [0, 255], тип uint8).

    Notes
    -----
    - Использует глобальные переменные `mean` и `std` для выполнения обратной нормализации.
    - Преобразует значения в диапазон [0, 255] и тип uint8.
    """
    # Преобразуем mean и std в тензоры и переносим их на то же устройство, что и image
    tensor_mean = torch.tensor(mean).view(-1, 1, 1).to(image_tensor.device)
    tensor_std = torch.tensor(std).view(-1, 1, 1).to(image_tensor.device)

    # Денормализация: (тензор * std) + mean
    denormalize_image = image_tensor * tensor_std + tensor_mean

    # Преобразуем в диапазон [0, 255] и к типу uint8
    return (denormalize_image * 255).clamp(0, 255).byte()


def show_images(dataset, amount=3, figsize=(4, 4), classes=None, n_classes=None):
    """
    Отображает изображения из датасета, организованные по классам.

    Parameters
    ----------
    dataset : ImageClassificationDataset
        Датасет, содержащий изображения и их метки.
    amount : int, optional
        Количество изображений для каждого класса (по умолчанию 3).
    figsize : tuple, optional
        Размер фигуры в формате (ширина, высота) для одного изображения (по умолчанию (4, 4)).
    classes : dict or list, optional
        Словарь или список, где индекс класса соответствует названию класса (по умолчанию None).
    n_classes : int, optional
        Максимальное количество классов для отображения (по умолчанию None, отображаются все классы).

    Notes
    -----
    - Предполагается, что датасет имеет атрибут `labels`, содержащий метки классов.
    - Для корректного отображения изображений используется функция `denormalize`.

    """
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
    """
    Базовый класс датасета для загрузки изображений с применением трансформаций.

    Parameters
    ----------
    image_paths : list
        Список путей к изображениям.
    transform : callable
        Трансформация, применяемая к изображениям.

    """
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        """
        Возвращает количество изображений в датасете.

        Returns
        -------
        int
            Количество изображений.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Возвращает изображение после применения трансформаций.

        Parameters
        ----------
        idx : int
            Индекс изображения в датасете.

        Returns
        -------
        torch.Tensor
            Преобразованное изображение в формате PyTorch Tensor.
        """
        # Считываем изображение
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Применяем трансформации
        image = self.transform(image)

        return image.to(device)


class ImageClassificationDataset(ImageDataset):
    """
    Датасет для классификации изображений, дополненный метками классов.

    Parameters
    ----------
    image_paths : list
        Список путей к изображениям.
    labels : list
        Список меток классов, соответствующих изображениям.
    transform : callable
        Трансформация, применяемая к изображениям.

    """
    def __init__(self, image_paths, labels, transform):
        super().__init__(image_paths, transform)
        self.labels = labels

    def __getitem__(self, idx):
        """
        Возвращает изображение и его метку.

        Parameters
        ----------
        idx : int
            Индекс изображения в датасете.

        Returns
        -------
        tuple
            torch.Tensor: Преобразованное изображение.
            torch.Tensor: Метка класса в формате PyTorch Tensor.
        """
        # Получаем метку
        label = self.labels[idx]

        return super().__getitem__(idx), torch.tensor(label, dtype=torch.long).to(device)
