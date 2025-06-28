import os
import torch
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn, optim
from collections import deque
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from IPython.display import clear_output
from sklearn.metrics import accuracy_score


class Classifier(nn.Module):
    def __init__(self, model, name='Model', optimizer=None, scheduler=None,
                 loss_fn=None, metric=None, model_dir=None, exist_ok=True):
        super().__init__()

        # Название модели
        self.name = name

        # Устройство для обучения
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        # Распараллеливаем модель по устройствам (CPU или GPU)
        self.__model = nn.DataParallel(model)

        # Инициализируем историю
        self.__lr_history = []
        self.__train_loss_history, self.__valid_loss_history = [], []
        self.__train_score_history, self.__valid_score_history = [], []

        # Лучшие значения
        self.best_score, self.best_score_epoch = None, 0
        self.best_loss, self.best_loss_epoch = None, 0

        # Флаг для остановки обучения
        self.stop_fiting = False

        if model_dir is None:
            model_dir = f"./models/{name}"
        self.model_dir = model_dir

        # Путь для сохранения модели
        if os.path.exists(model_dir):
            results_path = f"{self.model_dir}/results.csv"

            if exist_ok and os.path.exists(results_path) and os.listdir(f"{self.model_dir}/weights"):
                results = pd.read_csv(results_path)

                # Синхронизируем историю
                self.__lr_history = results['lr'].tolist()
                self.__train_loss_history, self.__valid_loss_history = results['train_loss'].tolist(), results['valid_loss'].tolist()
                self.__train_score_history, self.__valid_score_history = results['train_score'].tolist(), results['valid_score'].tolist()

                # Лучшие значения
                self.best_score, self.best_score_epoch = results['valid_score'].max(), results['valid_score'].argmax() + 1
                self.best_loss, self.best_loss_epoch = results['valid_loss'].min(), results['valid_loss'].argmin() + 1

                self.load("last")
            else:
                shutil.rmtree(model_dir)

        os.makedirs(f"{self.model_dir}/weights", exist_ok=True)

    # Property атрибуты (без записи)
    @property
    def loss_fn(self):
        return self.__loss_fn

    @property 
    def metric(self):
        return self.__metric

    @property
    def model(self):
        return self.__model

    @property
    def lr_history(self):
        return self.__lr_history

    @property
    def train_loss_history(self):
        return self.__train_loss_history

    @property
    def valid_loss_history(self):
        return self.__valid_loss_history

    @property
    def train_score_history(self):
        return self.__train_score_history

    @property
    def valid_score_history(self):
        return self.__valid_score_history

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

        y_maxlen = 1000
        y_true = deque(maxlen=y_maxlen)
        y_pred = deque(maxlen=y_maxlen)

        # Название для tqdm
        progress_desc = 'Training' if mode == 'train' else 'Evaluating'
        progress_bar = tqdm(data_loader, desc=progress_desc)
        display = dict()

        try:
            for batch in progress_bar:
                model_args = batch.pop('model_args', list())
                for i in range(len(model_args)):
                    model_args[i] = model_args[i].to(self.device)

                model_kwargs = batch.pop('model_kwargs', dict())
                for key in model_kwargs:
                    model_kwargs[key] = model_kwargs[key].to(self.device)

                labels = batch.pop('labels').to(self.device)

                if mode == 'train':
                    self.__optimizer.zero_grad()

                # Прямой проход
                output = self.__model(*model_args, **model_kwargs)
                loss = self.__loss_fn(output, labels)

                # Обратное распространение и шаг оптимизатора только в режиме тренировки
                if mode == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.__model.parameters(), 1.0)
                    self.__optimizer.step()

                    if isinstance(self.__scheduler, optim.lr_scheduler.OneCycleLR):
                        self.__scheduler.step()   # Делаем шаг для OneCycleLR
                        self.lr = self.__scheduler.get_last_lr()[0]
                        display['lr'] = round(self.lr, 10)

                # Подсчет потерь и метрик
                total_loss += loss.item()

                y_true.extend(labels.tolist())
                y_pred.extend(output.argmax(dim=1).tolist())

                count += 1

                # Обновляем описание tqdm с текущими значениями
                current_loss = total_loss / count
                current_score = self.__metric(np.array(y_true), np.array(y_pred))

                display.update({
                    self.__loss_fn.__class__.__name__: f"{current_loss:.4f}",
                    self.__metric.__name__: f"{current_score:.4f}"
                })

                progress_bar.set_postfix(**display)

        except KeyboardInterrupt:
            self.stop_fiting = True
            print(f"\n{progress_desc} прервано пользователем. Завершаем текущую эпоху...")

            if not count:
                return 0, 0

        # Возвращаем средний loss и метрику за эпоху
        return total_loss / count, current_score

    def plot_stats(self):
        # Создаем объект фигуры
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        epochs = range(1, len(self.__train_loss_history) + 1)

        # Визуализация потерь
        sns.lineplot(ax=axes[0], x=epochs, y=self.__train_loss_history, label='Train Loss', linestyle='--', marker='o',
                    color='#1f77b4', linewidth=3)
        sns.lineplot(ax=axes[0], x=epochs, y=self.__valid_loss_history, label='Valid Loss', linestyle='-', marker='o',
                    color='#bc4b51', linewidth=3)
        axes[0].plot(epochs, self.__valid_loss_history, 'o', markerfacecolor='none', markeredgecolor='#bc4b51', markersize=7,
                    linewidth=2)
        axes[0].set_title(f'{self.name} - {self.__loss_fn.__class__.__name__}')
        axes[0].set_xlabel('Epochs')
        axes[0].legend()
        axes[0].set_ylabel('')
        axes[0].set_xticks(epochs)
        axes[0].set_xlim(1, len(self.__train_loss_history))

        # Визуализация кастомной метрики
        sns.lineplot(ax=axes[1], x=epochs, y=self.__train_score_history, label=f'Train {self.__metric.__name__}', linestyle='--',
                    marker='o', linewidth=3)
        sns.lineplot(ax=axes[1], x=epochs, y=self.__valid_score_history, label=f'Valid {self.__metric.__name__}', linestyle='-',
                    marker='o', linewidth=3)
        axes[1].plot(epochs, self.__valid_score_history, 'o', markerfacecolor='none', markeredgecolor='#DD8452', markersize=7,
                    linewidth=2)
        axes[1].set_title(f'{self.name} - {self.__metric.__name__}')
        axes[1].set_xlabel('Epochs')
        axes[1].legend()
        axes[1].set_ylabel('')
        axes[1].set_xticks(epochs)
        axes[1].set_xlim(1, len(self.__train_score_history))

        fig.tight_layout()

        # Отображаем график
        plt.show()

        # Возвращаем объект фигуры
        return fig

    def fit(self, train_loader, valid_loader, num_epochs,
            min_loss=False, visualize=True, use_best_model=True, save_period=None):
        # Настраиваем стиль графиков
        sns.set_style('whitegrid')
        sns.set_palette('Set2')

        start_epoch = len(self.__train_loss_history) + 1

        if num_epochs < start_epoch:
            print(f"Модель уже обучена на {start_epoch - 1} эпох")
            return

        for epoch in range(start_epoch, num_epochs + 1):
            # Объявление о новой эпохе
            print(f"\nEpoch: {epoch}/{num_epochs}\n")

            # Обучение на тренировочных данных
            train_loss, train_score = self.run_epoch(train_loader, mode='train')

            # Оценка на валидационных данных
            valid_loss, valid_score = self.run_epoch(valid_loader, mode='eval')

            # Очищаем вывод для обновления информации
            clear_output()

            print(f"Epoch: {epoch}/{num_epochs}\n")

            print(f"Learning Rate: {round(self.lr, 10)}\n")

            print(f'Loss: {self.__loss_fn.__class__.__name__}')
            print(f" - Train: {train_loss:.4f}\n - Valid: {valid_loss:.4f}\n")

            print(f"Score: {self.__metric.__name__}")
            print(f" - Train: {train_score:.4f}\n - Valid: {valid_score:.4f}\n")

            if not self.stop_fiting:
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
                    "train_score": self.__train_score_history,
                    "valid_loss": self.__valid_loss_history,
                    "valid_score": self.__valid_score_history,
                }).to_csv(f"{self.model_dir}/results.csv", index=False)

                # Сохранение модели
                # - Last
                self.save("last")

                # - Best
                if self.best_loss is None or valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.best_loss_epoch = epoch

                    if min_loss and not self.stop_fiting:
                        self.save()

                if self.best_score is None or valid_score > self.best_score:
                    self.best_score = valid_score
                    self.best_score_epoch = epoch

                    if not min_loss and not self.stop_fiting:
                        self.save()

                # - Epoch
                if save_period is not None and epoch % save_period == 0:
                    self.save(epoch)

                # Делаем шаг планировщиком
                if self.__scheduler is not None and not isinstance(self.__scheduler, optim.lr_scheduler.OneCycleLR):
                    if isinstance(self.__scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.__scheduler.step(valid_loss if min_loss else valid_score)

                        if self.__scheduler.get_last_lr()[0] != self.lr:
                            self.load()
                    else:
                        self.__scheduler.step()
                    
                    self.lr = self.__scheduler.get_last_lr()[0]

            # Визуализация истории
            if len(self.__train_loss_history) > 1:
                if visualize:
                    # Сохранение графика в формате PNG
                    fig = self.plot_stats()
                    fig.savefig(f'{self.model_dir}/fiting_plot.png', dpi=300)

                print("Best:")
                print(f"Loss - {self.best_loss:.4f} ({self.best_loss_epoch} epoch)")
                print(f"Score - {self.best_score:.4f} ({self.best_score_epoch} epoch)\n")


            # Проверяем флаг остановки обучения
            if self.stop_fiting:
                print("Обучение остановлено пользователем после текущей эпохи.")

                self.stop_fiting = False
                break

        # Загружаем лучшие веса модели
        if use_best_model:
            self.load()

    @torch.inference_mode()
    def predict_proba(self, inputs, batch_size=10, num_workers=0, progress_bar=True):
        softmax = nn.Softmax(1)

        # Обработка одного объекта
        if isinstance(inputs, dict):            
            model_args = inputs.pop('model_args', list())
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device).unsqueeze(0)
            
            model_kwargs = inputs.pop('kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device).unsqueeze(0)
                
            return softmax(self.__model(*model_args, **model_kwargs)).squeeze().cpu().numpy()

        # Если формат данных неизвестен
        if not isinstance(inputs, torch.utils.data.Dataset):
            raise ValueError("Unsupported input type. Expected Dataset.")
        
        predictions = []
        data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if progress_bar:
            data_loader = tqdm(data_loader, desc="Predicting")

        # Итерация по батчам
        for batch in data_loader:
            if 'labels' in batch:
                batch.pop('labels')
                
            model_args = batch.pop('model_args', list())
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device)
                
            model_kwargs = batch.pop('model_kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device)
                
            batch_predictions = self.__model(*model_args, **model_kwargs)
            predictions.append(batch_predictions)

        return softmax(torch.cat(predictions, dim=0)).cpu().numpy()

    @torch.inference_mode()
    def predict(self, inputs, batch_size=10, num_workers=0, progress_bar=True):
        # Обработка одного объекта
        if isinstance(inputs, dict):
            if 'labels' in inputs:
                inputs = {k:v for k,v in inputs.items() if k != 'labels'}
            
            model_args = inputs.pop('model_args', list())            
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device).unsqueeze(0)
            
            model_kwargs = inputs.pop('model_kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device).unsqueeze(0)
                
            return np.argmax(self.__model(*model_args, **model_kwargs).squeeze().cpu().numpy())

        # Если формат данных неизвестен
        if not isinstance(inputs, torch.utils.data.Dataset):
            raise ValueError("Unsupported input type. Expected Dataset.")
        
        predictions = []
        data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if progress_bar:
            data_loader = tqdm(data_loader, desc="Predicting")

        # Итерация по батчам
        for batch in data_loader:
            if 'labels' in batch:
                batch.pop('labels')
                
            model_args = batch.pop('model_args', list())                
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device)
                
            model_kwargs = batch.pop('model_kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device)
                
            batch_predictions = self.__model(*model_args, **model_kwargs)
            predictions.append(np.argmax(batch_predictions.cpu().numpy(), axis=1))

        return np.hstack(predictions)

    def save(self, name="best", is_path=False):
        if not is_path:
            if isinstance(name, int) or name.isdigit():
                name = f"epoch_{name}"

            path = f"{self.model_dir}/weights/{name}.pt"
        else:
            path = name
        
        torch.save(self.__model.state_dict(), path)

    def load(self, name="best", is_path=False):
        if not is_path:
            if isinstance(name, int) or name.isdigit():
                name = f"epoch_{name}"

            path = f"{self.model_dir}/weights/{name}.pt"
        else:
            path = name

        if os.path.exists(path):
            # Переводим модель в режим оценки
            self.__model.eval()

            # Загружаем веса и применяем их к модели
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.__model.load_state_dict(state_dict)
        else:
            print(f"Не получилось загрузить модель, путь '{path}' не найден")
