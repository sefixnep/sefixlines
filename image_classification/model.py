# Работа с данными
import pandas as pd

# Torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Визуализация
import seaborn as sns

# Метрики
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Остальное
from tqdm.notebook import tqdm
from IPython.display import clear_output

# Configuration
from image_classification.data import *


def metrics(y_true, y_pred, count_round=4):
    """
    Вычисляет метрики качества классификации: accuracy, precision, recall и f1-score.

    Parameters
    ----------
    y_true : array-like
        Истинные метки классов.
    y_pred : array-like
        Предсказанные метки классов.
    count_round : int, optional
        Количество знаков после запятой для округления метрик (по умолчанию 4).

    Returns
    -------
    pd.Series
        Серия с рассчитанными метриками: accuracy, precision, recall и f1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    rounded_scores = np.round([accuracy, precision, recall, f1], count_round)
    index_labels = ['accuracy', 'precision', 'recall', 'f1']

    return pd.Series(rounded_scores, index=index_labels)


class ImageClassifier(nn.Module):
    """
    Класс для обучения, оценки и сохранения модели классификации изображений.

    Parameters
    ----------
    model : nn.Module
        PyTorch модель для классификации.
    name : str, optional
        Имя модели для сохранения файлов (по умолчанию 'Model').
    optimizer : torch.optim.Optimizer, optional
        Оптимизатор для обучения модели (по умолчанию Adam с lr=3e-4).
    loss_fn : callable, optional
        Функция потерь (по умолчанию CrossEntropyLoss).
    metric : callable, optional
        Метрика для оценки качества модели (по умолчанию accuracy_score).
    save_dir : str, optional
        Директория для сохранения модели (по умолчанию "../models").
    """
    def __init__(self, model, name='Model', optimizer=None, loss_fn=None, metric=None, save_dir="../models"):
        super().__init__()

        # Название модели
        self.name = name

        # Путь для сохранения модели
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.path = f"{save_dir}/{self.name}.pth"

        # Оптимизатор
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.__optimizer = optimizer

        # Функция потерь
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        self.__loss_fn = loss_fn

        # Метрика
        if metric is None:
            metric = accuracy_score
        self.__metric = metric

        # Переносим модель на устройство (CPU или GPU)
        self.__model = model.to(device)

        # Инициализируем историю качества
        self.__train_loss_history, self.__valid_loss_history = [], []
        self.__train_score_history, self.__valid_score_history = [], []

        # Лучшие значения
        self.best_epoch, self.best_score, self.best_loss = 0, 0, float('inf')

        # Флаг для остановки обучения
        self.stop_fiting = False

    def forward(self, x):
        """
        Прямой проход данных через модель.

        Parameters
        ----------
        x : torch.tensor
            Входные данные.

        Returns
        -------
        torch.tensor
            Результат работы модели.
        """
        return self.__model(x)

    def run_epoch(self, data_loader, mode='train'):
        """
        Выполняет одну эпоху обучения или оценки.

        Args:
            data_loader: DataLoader с данными.
            mode: Режим выполнения ('train' или 'eval').

        Returns:
            Средний loss и метрика за эпоху.
        """
        # Установка режима работы модели
        if mode == 'train':
            self.__model.train()
        elif mode == 'eval':
            self.__model.eval()
        else:
            raise ValueError("Mode должен быть 'train' или 'eval'.")

        # Переменные для подсчета
        count = 0
        total_loss = 0
        labels_true, labels_pred = [], []

        # Отключаем градиенты в режиме оценки
        torch.set_grad_enabled(mode == 'train')

        # Название для tqdm
        progress_desc = 'Training' if mode == 'train' else 'Evaluating'
        progress_bar = tqdm(data_loader, desc=progress_desc)

        try:
            for images, labels in progress_bar:
                if mode == 'train':
                    self.__optimizer.zero_grad()

                # Прямой проход
                output = self.__model(images)
                loss = self.__loss_fn(output, labels)

                # Обратное распространение и шаг оптимизатора только в режиме тренировки
                if mode == 'train':
                    loss.backward()
                    self.__optimizer.step()

                # Подсчет потерь и метрик
                total_loss += loss.item()
                labels_true.extend(labels.cpu().numpy())
                labels_pred.extend(output.argmax(dim=1).cpu().numpy())

                count += 1

                # Обновляем описание tqdm с текущими значениями
                current_loss = total_loss / count
                current_score = self.__metric(labels_true, labels_pred)
                progress_bar.set_postfix(
                    **{
                        self.__loss_fn.__class__.__name__: f"{current_loss:.4f}",
                        self.__metric.__name__: f"{current_score:.4f}",
                    }
                )

        except KeyboardInterrupt:
            self.stop_fiting = True
            print(f"\n{progress_desc} прервано пользователем. Завершаем текущую эпоху...")

        # Возвращаем средний loss и метрику за эпоху
        return total_loss / count, self.__metric(labels_true, labels_pred)

    def plot_stats(self):
        """
        Визуализирует историю потерь и метрик за все эпохи обучения.
        """
        # Настраиваем график
        plt.figure(figsize=(16, 8))
        epochs = range(1, len(self.__train_loss_history) + 1)

        # Визуализация потерь
        plt.subplot(1, 2, 1)

        sns.lineplot(x=epochs, y=self.__train_loss_history, label='Train Loss', linestyle='--', marker='o',
                     color='#1f77b4',
                     linewidth=3)
        sns.lineplot(x=epochs, y=self.__valid_loss_history, label='Valid Loss', linestyle='-', marker='o',
                     color='#bc4b51',
                     linewidth=3)
        plt.plot(epochs, self.__valid_loss_history, 'o', markerfacecolor='none', markeredgecolor='#bc4b51', markersize=7,
                 linewidth=2)

        plt.title(f'{self.name} - {self.__loss_fn.__class__.__name__}')
        plt.xlabel('Epochs')
        plt.legend()
        plt.gca().set_ylabel('')
        plt.xticks(epochs)  # Устанавливаем натуральные значения по оси x
        plt.xlim(1, len(self.__train_loss_history))  # Ограничиваем ось x от 1 до максимального значения

        # Визуализация кастомной метрики
        plt.subplot(1, 2, 2)

        sns.lineplot(x=epochs, y=self.__train_score_history, label=f'Train {self.__metric.__name__}', linestyle='--',
                     marker='o',
                     linewidth=3)
        sns.lineplot(x=epochs, y=self.__valid_score_history, label=f'Valid {self.__metric.__name__}', linestyle='-',
                     marker='o',
                     linewidth=3)
        plt.plot(epochs, self.__valid_score_history, 'o', markerfacecolor='none', markeredgecolor='#DD8452', markersize=7,
                 linewidth=2)

        plt.title(f'{self.name} - {self.__metric.__name__}')
        plt.xlabel('Epochs')
        plt.legend()
        plt.gca().set_ylabel('')
        plt.xticks(epochs)  # Устанавливаем натуральные значения по оси x
        plt.xlim(1, len(self.__train_score_history))  # Ограничиваем ось x от 1 до максимального значения

        plt.tight_layout()
        plt.show()

    def fit(self, train_loader, valid_loader, num_epochs,
            min_loss=False, visualize=True, use_best_model=True, eps=0.001):
        """
        Обучает модель на тренировочном наборе данных и оценивает на валидационном.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader для тренировочных данных.
        valid_loader : DataLoader
            DataLoader для валидационных данных.
        num_epochs : int
            Количество эпох обучения.
        min_loss : bool, optional
            Если True, сохраняет модель с минимальной потерей; иначе — с максимальной метрикой (по умолчанию False).
        visualize : bool, optional
            Если True, визуализирует графики после каждой эпохи (по умолчанию True).
        use_best_model : bool, optional
            Если True, загружает лучшие веса модели после завершения обучения (по умолчанию True).
        eps : float, optional
            Минимальное изменение метрики для обновления модели (по умолчанию 0.001).
        """
        # Настраиваем стиль графиков
        sns.set_style('whitegrid')
        sns.set_palette('Set2')

        for epoch in range(1, num_epochs + 1):
            # Объявление о новой эпохе
            print(f"\nEpoch: {epoch}/{num_epochs} (total: {len(self.__train_loss_history) + 1})\n")

            # Обучение на тренировочных данных
            train_loss, train_score = self.run_epoch(train_loader, mode='train')

            # Оценка на валидационных данных
            valid_loss, valid_score = self.run_epoch(valid_loader, mode='eval')

            # Сохраняем историю потерь и метрик
            self.__train_loss_history.append(train_loss)
            self.__valid_loss_history.append(valid_loss)
            self.__train_score_history.append(train_score)
            self.__valid_score_history.append(valid_score)

            # Очищаем вывод для обновления информации
            clear_output()

            print(f"Epoch: {epoch}/{num_epochs} (total: {len(self.__train_loss_history)})\n")
            print(f'Loss: {self.__loss_fn.__class__.__name__}')
            print(f" - Train: {train_loss:.4f}\n - Valid: {valid_loss:.4f}\n")

            print(f"Score: {self.__metric.__name__}")
            print(f" - Train: {train_score:.4f}\n - Valid: {valid_score:.4f}\n")

            # Сохраняем лучшую модель на основе улучшения метрики
            if (self.best_score is None or (valid_score - self.best_score > eps) or min_loss) and (
                    not min_loss or valid_loss < self.best_loss) and not self.stop_fiting:
                print("(Model saved)")
                self.best_epoch, self.best_score, self.best_loss = len(self.__train_loss_history), valid_score, valid_loss
                self.save_model()

            # Визуализация результатов после второй эпохи
            if len(self.__train_loss_history) > 1:
                if visualize:
                    self.plot_stats()

                print(f"Best valid score: {self.best_score:.4f} ({self.best_epoch} epoch)\n")

            # Проверяем флаг остановки обучения
            if self.stop_fiting:
                print("Обучение остановлено пользователем после текущей эпохи.")

                self.stop_fiting = False
                break

        # Загружаем лучшие веса модели
        if use_best_model and os.path.exists(self.path):
            # Losses
            self.__train_loss_history = self.__train_loss_history[:self.best_epoch]
            self.__valid_loss_history = self.__valid_loss_history[:self.best_epoch]
            # Scores
            self.__train_score_history = self.__train_score_history[:self.best_epoch]
            self.__valid_score_history = self.__valid_score_history[:self.best_epoch]

            # Load best model
            self.load()

    @torch.inference_mode()
    def predict_proba(self, inputs, batch_size=50, progress_bar=True):
        """
        Предсказывает вероятности классов для входных данных.

        Parameters
        ----------
        inputs : torch.Tensor, list или Dataset
            Входные данные для предсказания.
        batch_size : int, optional
            Размер батча для предсказания (по умолчанию 50).
        progress_bar : bool, optional
            Если True, отображает прогресс-бар (по умолчанию True).

        Returns
        -------
        list
            Список предсказанных вероятностей классов.
        """
        # Обработка одного изображения
        if isinstance(inputs, torch.Tensor) and inputs.dim() == 3:
            return self.__model(inputs.unsqueeze(0).to(device))[0].tolist()

        # Определяем, является ли входной списоком или датасетом
        if isinstance(inputs, (list, Dataset)):
            predictions = []
            data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False)

            if progress_bar:
                data_loader = tqdm(data_loader, desc="Predicting")

            # Итерация по батчам
            for batch in data_loader:
                batch_predictions = self.__model(batch.to(device))
                predictions.append(batch_predictions)

            return torch.cat(predictions, dim=0).tolist()

        # Если формат данных неизвестен
        raise ValueError("Unsupported input type. Expected single tensor, list of tensors, or Dataset.")

    def predict(self, inputs, *args, **kwargs):
        """
        Предсказывает классы для входных данных.

        Parameters
        ----------
        inputs : torch.Tensor, list или Dataset
            Входные данные для предсказания.
        *args, **kwargs : остальные значения для метода self.predict_proba

        Returns
        -------
        list
            Список предсказанных меток классов.
        """
        return np.argmax(self.predict_proba(inputs, *args, **kwargs), axis=1
            if isinstance(inputs, (list, Dataset)) else None).tolist()

    def save_model(self):
        """
        Сохраняет текущие веса модели в файл.
        """
        torch.save(self.__model.state_dict(), self.path)

    def load(self):
        """
        Загружает веса модели из сохраненного файла.
        """
        # Переводим модель в режим оценки
        self.__model.eval()

        # Загружаем веса и применяем их к модели
        state_dict = torch.load(self.path, map_location=device, weights_only=True)
        self.__model.load_state_dict(state_dict)
