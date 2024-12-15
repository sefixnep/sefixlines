# Работа с данными
import pandas as pd

# Torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Визуализация
import seaborn as sns

# Метрики
from sklearn.metrics import accuracy_score

# Остальное
import shutil
from IPython.display import clear_output

# Configuration
from classification.data import *


class Classifier(nn.Module):
    def __init__(self, model, name='Model', optimizer=None, scheduler=None,
                 loss_fn=None, metric=None, model_dir=None, exist_ok=False):
        super().__init__()

        # Название модели
        self.name = name

        if model_dir is None:
            model_dir = f"./models/{name}"
        self.model_dir = model_dir

        # Путь для сохранения модели
        if os.path.exists(model_dir):
            assert exist_ok, "Папка с моделью уже занята"
            shutil.rmtree(model_dir)

        os.makedirs(self.model_dir)
        os.makedirs(f"{self.model_dir}/weights")

        # Оптимизатор
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.__optimizer = optimizer

        # Планировщик
        self.__scheduler = scheduler
        self.lr = optimizer.param_groups[0]['lr']

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

        # Инициализируем историю
        self.__lr_history = []
        self.__train_loss_history, self.__valid_loss_history = [], []
        self.__train_score_history, self.__valid_score_history = [], []

        # Лучшие значения
        self.best_score, self.best_score_epoch = None, 0
        self.best_loss, self.best_loss_epoch = None, 0

        # Флаг для остановки обучения
        self.stop_fiting = False

    def forward(self, x):
        return self.__model(x)

    def run_epoch(self, data_loader, mode='train'):
        # Установка режима работы модели
        if mode == 'train':
            self.__model.train()
        elif mode == 'eval':
            self.__model.eval()
        else:
            raise ValueError("Mode должен быть 'train' или 'eval'.")
        
        # Отключаем градиенты в режиме оценки
        torch.set_grad_enabled(mode == 'train')

        # Переменные для подсчета
        count = 0
        total_loss = 0
        total_score = 0

        # Название для tqdm
        progress_desc = 'Training' if mode == 'train' else 'Evaluating'
        progress_bar = tqdm(data_loader, desc=progress_desc)

        try:
            for data, labels in progress_bar:
                if mode == 'train':
                    self.__optimizer.zero_grad()

                # Прямой проход
                output = self.__model(data)
                loss = self.__loss_fn(output, labels)

                # Обратное распространение и шаг оптимизатора только в режиме тренировки
                if mode == 'train':
                    loss.backward()
                    self.__optimizer.step()

                labels_true = labels.cpu().numpy()
                labels_pred = output.argmax(dim=1).cpu().numpy()

                # Подсчет потерь и метрик
                total_loss += loss.item()
                total_score += self.__metric(labels_true, labels_pred)

                count += 1

                # Обновляем описание tqdm с текущими значениями
                current_loss = total_loss / count
                current_score = total_score / count
                progress_bar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    **{self.__metric.__name__: f"{current_score:.4f}"}
                )

        except KeyboardInterrupt:
            self.stop_fiting = True
            print(f"\n{progress_desc} прервано пользователем. Завершаем текущую эпоху...")

            if not count:
                return 0, 0

        # Возвращаем средний loss и метрику за эпоху
        return total_loss / count, total_score / count

    def plot_stats(self):
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
            min_loss=False, visualize=True, use_best_model=True, save_period=None):
        # Настраиваем стиль графиков
        sns.set_style('whitegrid')
        sns.set_palette('Set2')

        for epoch in range(len(self.__train_loss_history) + 1, len(self.__train_loss_history) + num_epochs + 1):
            # Объявление о новой эпохе
            print(f"\nEpoch: {epoch}/{num_epochs} (total: {len(self.__train_loss_history) + 1})\n")

            # Обучение на тренировочных данных
            train_loss, train_score = self.run_epoch(train_loader, mode='train')

            # Оценка на валидационных данных
            valid_loss, valid_score = self.run_epoch(valid_loader, mode='eval')

            # Очищаем вывод для обновления информации
            clear_output()

            print(f"Epoch: {epoch}/{num_epochs} (total: {len(self.__train_loss_history) + 1})\n")

            print(f"Learning Rate: {self.lr}\n")

            print(f'Loss: {self.__loss_fn.__class__.__name__}')
            print(f" - Train: {train_loss:.4f}\n - Valid: {valid_loss:.4f}\n")

            print(f"Score: {self.__metric.__name__}")
            print(f" - Train: {train_score:.4f}\n - Valid: {valid_score:.4f}\n")

            # Сохранение истории
            self.__lr_history.append(self.lr)
            self.__train_loss_history.append(train_loss)
            self.__valid_loss_history.append(valid_loss)
            self.__train_score_history.append(train_score)
            self.__valid_score_history.append(valid_score)

            pd.DataFrame({
                "epoch": range(1, len(self.__train_loss_history) + 1),
                "lr": self.__lr_history,
                "train_loss": self.__train_loss_history,
                "valid_loss": self.__valid_loss_history,
                "train_score": self.__train_score_history,
                "valid_score": self.__valid_score_history,
            }).to_csv(f"{self.model_dir}/results.csv", index=False)

            # Визуализация истории
            if len(self.__train_loss_history) > 1:
                if visualize:
                    self.plot_stats()

                print("Best:")
                print(f"Loss - {self.best_loss:.4f} ({self.best_loss_epoch} epoch)")
                print(f"Score - {self.best_score:.4f} ({self.best_score_epoch} epoch)\n")

            # Сохранение модели
            # - Last
            self.save_model("last")

            # - Best
            if self.best_loss is None or valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_loss_epoch = epoch

                if min_loss and not self.stop_fiting:
                    self.save_model()

            if self.best_score is None or valid_score > self.best_score:
                self.best_score = valid_score
                self.best_score_epoch = epoch

                if not min_loss and not self.stop_fiting:
                    self.save_model()

            # - Epoch
            if save_period is not None and epoch % save_period == 0:
                self.save_model(epoch)

            # Делаем шаг планировщиком
            if self.__scheduler is not None:
                self.__scheduler.step()
                self.lr = self.__scheduler.get_last_lr()[0]

            # Проверяем флаг остановки обучения
            if self.stop_fiting:
                print("Обучение остановлено пользователем после текущей эпохи.")

                self.stop_fiting = False
                break

        # Загружаем лучшие веса модели
        if use_best_model:
            self.load()

    @torch.inference_mode()
    def predict_proba(self, inputs, batch_size=10, progress_bar=True):
        # Обработка одного изображения
        if isinstance(inputs, torch.Tensor) and inputs.dim() == 3:
            return self.__model(inputs.unsqueeze(0).to(device))[0].tolist()

        # Если формат данных неизвестен
        if not isinstance(inputs, Dataset):
            raise ValueError("Unsupported input type. Expected single tensor, list of tensors, or Dataset.")
        
        predictions = []
        data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False)

        if progress_bar:
            data_loader = tqdm(data_loader, desc="Predicting")

        # Итерация по батчам
        for batch in data_loader:
            batch_predictions = self.__model(batch.to(device))
            predictions.append(batch_predictions)

        return torch.cat(predictions, dim=0).tolist()

    def predict(self, inputs, *args, **kwargs):
        return np.argmax(self.predict_proba(inputs, *args, **kwargs), axis=1
            if isinstance(inputs, (list, Dataset)) else None).tolist()

    def save_model(self, name="best"):
        if isinstance(name, int) or name.isdigit():
            name = f"epoch_{name}"

        path = f"{self.model_dir}/weights/{name}.pth"
        torch.save(self.__model.state_dict(), path)

    def load(self, name="best"):
        if isinstance(name, int) or name.isdigit():
            name = f"epoch_{name}"

        path = f"{self.model_dir}/weights/{name}.pth"
        if os.path.exists(path):
            # Переводим модель в режим оценки
            self.__model.eval()

            # Загружаем веса и применяем их к модели
            state_dict = torch.load(path, map_location=device, weights_only=True)
            self.__model.load_state_dict(state_dict)
        else:
            print(f"Не получилось загрузить модель, путь '{path}' не найден")
