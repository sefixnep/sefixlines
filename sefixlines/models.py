import os
import torch
import random
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch import nn, optim
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from IPython.display import clear_output
from sklearn.metrics import accuracy_score


class BaseModel(nn.Module):
    """Базовый класс для моделей с основной архитектурой для обучения."""
    
    def __init__(self, model, name='Model', optimizer=None, scheduler=None,
                 loss_fn=None, metric=None, model_dir=None, exist_ok=True, target='labels'):
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
            loss_fn = self._default_loss_fn()
        self.__loss_fn = loss_fn

        # Метрика
        if metric is None:
            metric = self._default_metric()
        self.__metric = metric
        
        # Тип ответа (для совместимости с разными задачами)
        self.target = target

        # Переносим модель на устройство
        self.__model = model.to(self.device)

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
                # Проверяем, есть ли валидационные данные (не все NaN)
                if not results['valid_score'].isna().all():
                    self.best_score, self.best_score_epoch = results['valid_score'].max(), results['valid_score'].idxmax() + 1
                    self.best_loss, self.best_loss_epoch = results['valid_loss'].min(), results['valid_loss'].idxmin() + 1
                else:
                    # Используем тренировочные метрики
                    self.best_score, self.best_score_epoch = results['train_score'].max(), results['train_score'].idxmax() + 1
                    self.best_loss, self.best_loss_epoch = results['train_loss'].min(), results['train_loss'].idxmin() + 1

                self.load("last")
            else:
                shutil.rmtree(model_dir)

        os.makedirs(f"{self.model_dir}/weights", exist_ok=True)
    
    def _default_loss_fn(self, *args, **kwargs):
        """Метод должен быть переопределен в дочерних классах."""
        raise NotImplementedError("Subclasses must implement _default_loss_fn")
    
    def _default_metric(self, *args, **kwargs):
        """Метод должен быть переопределен в дочерних классах."""
        raise NotImplementedError("Subclasses must implement _default_metric")

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

    def run_epoch(self, data_loader, mode='train', **kwargs):
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
        display = dict()

        try:
            for batch in progress_bar:
                model_args = batch.pop('model_args', list())
                for i in range(len(model_args)):
                    model_args[i] = model_args[i].to(self.device)

                model_kwargs = batch.pop('model_kwargs', dict())
                for key in model_kwargs:
                    model_kwargs[key] = model_kwargs[key].to(self.device)

                target = batch.pop(self.target).to(self.device)

                if mode == 'train':
                    self.__optimizer.zero_grad()

                # Прямой проход
                output = self.__model(*model_args, **model_kwargs)
                loss = self.__loss_fn(output, target)

                # Обратное распространение и шаг оптимизатора только в режиме тренировки
                if mode == 'train':
                    loss.backward()
                    if 'clip_grad_norm' in kwargs:
                        torch.nn.utils.clip_grad_norm_(self.__model.parameters(), kwargs['clip_grad_norm'])
                    self.__optimizer.step()

                    if isinstance(self.__scheduler, optim.lr_scheduler.OneCycleLR):
                        self.__scheduler.step()   # Делаем шаг для OneCycleLR
                        self.lr = self.__scheduler.get_last_lr()[0]
                        display['lr'] = round(self.lr, 10)

                # Подсчет потерь и метрик с учетом специализированной логики
                if getattr(self, 'multi_label', False):
                    prediction = torch.sigmoid(output).round()
                else:
                    prediction = output.argmax(dim=1)

                total_loss += loss.item()
                total_score += self.__metric(prediction.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())
                count += 1

                # Обновляем описание tqdm с текущими значениями
                current_loss = total_loss / count
                current_score = total_score / count

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
        return current_loss, current_score

    def plot_stats(self):
        # Создаем объект фигуры
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        epochs = range(1, len(self.__train_loss_history) + 1)

        # Проверяем, есть ли валидационные данные
        has_valid_data = not all(np.isnan(self.__valid_loss_history))

        # Визуализация потерь
        sns.lineplot(ax=axes[0], x=epochs, y=self.__train_loss_history, label='Train Loss', linestyle='--', marker='o',
                    color='#1f77b4', linewidth=3)
        if has_valid_data:
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
        if has_valid_data:
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

    def fit(self, 
            train_loader: DataLoader, valid_loader: DataLoader = None, num_epochs: int = 1,
            min_loss: bool = False, visualize: bool = True, use_best_model: bool = True, save_period: int = None, 
            train_params: dict = dict(), eval_params: dict = dict()
        ):
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
            train_loss, train_score = self.run_epoch(train_loader, mode='train', **train_params)

            # Оценка на валидационных данных (если есть валидационный набор)
            if valid_loader is not None:
                valid_loss, valid_score = self.run_epoch(valid_loader, mode='eval', **eval_params)
            else:
                valid_loss, valid_score = None, None

            # Очищаем вывод для обновления информации
            clear_output()

            print(f"Epoch: {epoch}/{num_epochs}\n")

            print(f"Learning Rate: {round(self.lr, 10)}\n")

            print(f'Loss: {self.__loss_fn.__class__.__name__}')
            if valid_loss is not None:
                print(f" - Train: {train_loss:.4f}\n - Valid: {valid_loss:.4f}\n")
            else:
                print(f" - Train: {train_loss:.4f}\n")

            print(f"Score: {self.__metric.__name__}")
            if valid_score is not None:
                print(f" - Train: {train_score:.4f}\n - Valid: {valid_score:.4f}\n")
            else:
                print(f" - Train: {train_score:.4f}\n")

            if not self.stop_fiting:
                # Сохранение истории
                self.__lr_history.append(self.lr)
                self.__train_loss_history.append(train_loss)
                self.__valid_loss_history.append(valid_loss if valid_loss is not None else float('nan'))
                self.__train_score_history.append(train_score)
                self.__valid_score_history.append(valid_score if valid_score is not None else float('nan'))

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

                # - Best (только если есть валидационные данные)
                if valid_loader is not None:
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
                else:
                    # Если нет валидации, сохраняем на основе тренировочных метрик
                    if self.best_loss is None or train_loss < self.best_loss:
                        self.best_loss = train_loss
                        self.best_loss_epoch = epoch

                        if min_loss and not self.stop_fiting:
                            self.save()

                    if self.best_score is None or train_score > self.best_score:
                        self.best_score = train_score
                        self.best_score_epoch = epoch

                        if not min_loss and not self.stop_fiting:
                            self.save()

                # - Epoch
                if save_period is not None and epoch % save_period == 0:
                    self.save(epoch)

                # Делаем шаг планировщиком
                if self.__scheduler is not None and not isinstance(self.__scheduler, optim.lr_scheduler.OneCycleLR):
                    if isinstance(self.__scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if valid_loader is not None:
                            self.__scheduler.step(valid_loss if min_loss else valid_score)
                        else:
                            self.__scheduler.step(train_loss if min_loss else train_score)

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

                if valid_loader is not None:
                    print("Best:")
                    print(f"Loss - {self.best_loss:.4f} ({self.best_loss_epoch} epoch)")
                    print(f"Score - {self.best_score:.4f} ({self.best_score_epoch} epoch)\n")
                else:
                    print("Best (train):")
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
    def predict_proba(self, inputs, batch_size=10, num_workers=0, progress_bar=True, apply_activation=True):
        self.__model.eval()

        # Если формат данных неизвестен
        if not isinstance(inputs, (dict, torch.utils.data.Dataset)):
            raise ValueError("Unsupported input type. Expected dict or Dataset.")

        # Обработка одного объекта
        if isinstance(inputs, dict):            
            model_args = inputs.pop('model_args', list())
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device).unsqueeze(0)
            
            model_kwargs = inputs.pop('model_kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device).unsqueeze(0)
                
            output = self.__model(*model_args, **model_kwargs).squeeze()
            if apply_activation:
                if getattr(self, 'multi_label', False):
                    output = torch.sigmoid(output)
                else:
                    output = output.softmax(dim=0)
            return output.cpu().numpy()
        
        predictions = []
        data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if progress_bar:
            data_loader = tqdm(data_loader, desc="Predicting probabilities")

        # Итерация по батчам
        for batch in data_loader:
            if self.target in batch:
                batch.pop(self.target)
                
            model_args = batch.pop('model_args', list())
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device)
                
            model_kwargs = batch.pop('model_kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device)
                
            output = self.__model(*model_args, **model_kwargs)
            if apply_activation:
                if getattr(self, 'multi_label', False):
                    output = torch.sigmoid(output)
                else:
                    output = output.softmax(dim=1)
            predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    @torch.inference_mode()
    def predict(self, inputs, batch_size=10, num_workers=0, progress_bar=True, threshold=0.5):
        self.__model.eval()

        # Если формат данных неизвестен
        if not isinstance(inputs, (dict, torch.utils.data.Dataset)):
            raise ValueError("Unsupported input type. Expected dict or Dataset.")
        
        # Обработка одного объекта
        if isinstance(inputs, dict):
            if self.target in inputs:
                inputs = {k:v for k,v in inputs.items() if k != self.target}
            
            model_args = inputs.pop('model_args', list())            
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device).unsqueeze(0)
            
            model_kwargs = inputs.pop('model_kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device).unsqueeze(0)
                
            output = self.__model(*model_args, **model_kwargs).squeeze()
            if getattr(self, 'multi_label', False):
                return (torch.sigmoid(output) > threshold).cpu().numpy().astype(int)
            else:
                return output.argmax(dim=0).cpu().numpy()
        
        predictions = []
        data_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if progress_bar:
            data_loader = tqdm(data_loader, desc="Predicting")

        # Итерация по батчам
        for batch in data_loader:
            if self.target in batch:
                batch.pop(self.target)
                
            model_args = batch.pop('model_args', list())                
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device)
                
            model_kwargs = batch.pop('model_kwargs', dict())
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].to(self.device)
                
            output = self.__model(*model_args, **model_kwargs)
            if getattr(self, 'multi_label', False):
                predictions.append((torch.sigmoid(output) > threshold).cpu().numpy().astype(int))
            else:
                predictions.append(output.argmax(dim=1).cpu().numpy())

        return np.concatenate(predictions, axis=0)

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


class Classifier(BaseModel):
    """Классификатор, наследуется от BaseModel."""
    
    def __init__(self, model, name='Classifier', optimizer=None, scheduler=None,
                 loss_fn=None, metric=None, model_dir=None, exist_ok=True, target='labels', multi_label=False):
        self.multi_label = multi_label
        super().__init__(model, name, optimizer, scheduler, loss_fn, metric, model_dir, exist_ok, target)
    
    def _default_loss_fn(self):
        """Функция потерь по умолчанию для классификации."""
        return nn.CrossEntropyLoss() if not self.multi_label else nn.BCEWithLogitsLoss()
    
    def _default_metric(self):
        """Метрика по умолчанию для классификации."""
        return accuracy_score

    def visualize_predictions(self, dataset, n=(2, 4), classes=None, fig_image_size=5, threshold=0.5):        
        # Определяем тип датасета
        is_image = hasattr(dataset, 'image_paths')
        
        if is_image:
            # Визуализация для изображений
            fig, axes = plt.subplots(n[0], n[1], figsize=(fig_image_size * n[1], fig_image_size * n[0]))
            
            # Обрабатываем случай, когда axes не двумерный
            if n[0] == 1 and n[1] == 1:
                axes = np.array([[axes]])
            elif n[0] == 1:
                axes = axes.reshape(1, -1)
            elif n[1] == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(n[0]):
                for j in range(n[1]):
                    idx = random.randrange(len(dataset))
                    batch = dataset[idx]
                    item = dataset.get_item(idx)
                    image = item['image'].resize((512, 512))
                    
                    # Предсказание
                    prediction = self.predict(batch, progress_bar=False, threshold=threshold)
                    
                    # Истинная метка
                    true_label = item.get('label')
                    
                    # Формируем текст
                    if self.multi_label:
                        # Multi-label
                        if classes is None:
                            true_str = ', '.join([str(i) for i, v in enumerate(true_label) if v > 0.5]) if true_label is not None else 'N/A'
                            pred_str = ', '.join([str(i) for i, v in enumerate(prediction) if v > 0.5])
                        else:
                            true_str = ', '.join([classes[i] for i, v in enumerate(true_label) if v > 0.5]) if true_label is not None else 'N/A'
                            pred_str = ', '.join([classes[i] for i, v in enumerate(prediction) if v > 0.5])
                    else:
                        # Single-label
                        if classes is None:
                            true_str = str(true_label) if true_label is not None else 'N/A'
                            pred_str = str(prediction)
                        else:
                            true_str = classes[true_label] if true_label is not None else 'N/A'
                            pred_str = classes[prediction]
                    
                    # Отображение изображения
                    ax = axes[i][j]
                    ax.imshow(np.array(image))
                    ax.axis('off')
                    ax.set_title(f"Class: {true_str}\nPredict: {pred_str}", fontsize=10)
            
            plt.subplots_adjust(hspace=0.3)
            plt.tight_layout()
            plt.show()
                
        else:
            # Визуализация для текста
            total_amount = n[0] * n[1] if isinstance(n, tuple) else n
            indices = random.sample(range(len(dataset)), min(total_amount, len(dataset)))
            
            for idx in indices:
                batch = dataset[idx]
                item = dataset.get_item(idx)
                text = item['text'].replace('\n', ' \\n ')
                
                # Предсказание
                prediction = self.predict(batch, progress_bar=False, threshold=threshold)
                
                # Истинная метка
                true_label = item.get('label')
                
                # Формируем текст
                if self.multi_label:
                    # Multi-label
                    if classes is None:
                        true_str = ', '.join([str(i) for i, v in enumerate(true_label) if v > 0.5]) if true_label is not None else 'N/A'
                        pred_str = ', '.join([str(i) for i, v in enumerate(prediction) if v > 0.5])
                    else:
                        true_str = ', '.join([classes[i] for i, v in enumerate(true_label) if v > 0.5]) if true_label is not None else 'N/A'
                        pred_str = ', '.join([classes[i] for i, v in enumerate(prediction) if v > 0.5])
                else:
                    # Single-label
                    if classes is None:
                        true_str = str(true_label) if true_label is not None else 'N/A'
                        pred_str = str(prediction)
                    else:
                        true_str = classes[true_label] if true_label is not None else 'N/A'
                        pred_str = classes[prediction]
                
                # Выводим
                text_preview = text[:80] + '...' if len(text) > 80 else text
                print(f"Text: {text_preview}")
                print(f"Class: {true_str}")
                print(f"Predict: {pred_str}")
                print()


class SemanticSegmenter(BaseModel):
    """Semantic Segmenter, наследуется от BaseModel."""
    
    def __init__(self, model, name='SemanticSegmenter', optimizer=None, scheduler=None,
                 loss_fn=None, metric=None, model_dir=None, exist_ok=True, target='masks'):
        super().__init__(model, name, optimizer, scheduler, loss_fn, metric, model_dir, exist_ok, target)
    
    def _default_loss_fn(self):
        """Функция потерь по умолчанию для семантической сегментации."""
        return smp.losses.DiceLoss(mode='multiclass')
    
    def _default_metric(self):
        """Метрика по умолчанию для семантической сегментации."""

        def iou_score(target, prediction):
            num_classes = max(target.max(), prediction.max()) + 1
            ious = []
            for cls in range(num_classes):
                target_cls = (target == cls)
                pred_cls = (prediction == cls)
                intersection = np.logical_and(target_cls, pred_cls).sum()
                union = np.logical_or(target_cls, pred_cls).sum()
                if union == 0:
                    ious.append(np.nan)
                else:
                    ious.append(intersection / union)
            # Среднее по всем классам, игнорируя nan
            return np.nanmean(ious)
        
        return iou_score

    def visualize_segmentation(self, dataset, amount=3, figsize=(5, 5)):        
        rows = amount
        cols = 3  # Оригинальное изображение, предсказанная маска, оригинальная маска
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
        
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        
        indices = random.sample(range(len(dataset)), min(amount, len(dataset)))
        
        for row, idx in enumerate(indices):
            # Получаем данные
            batch = dataset[idx]
            item = dataset.get_item(idx)
            image = item['image'].resize((512, 512))
            
            # Предсказание маски
            # Получаем raw output от модели
            model_args = batch.pop('model_args', list())
            for i in range(len(model_args)):
                model_args[i] = model_args[i].to(self.device).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(*model_args)
                predicted_mask = output.argmax(dim=1).squeeze().cpu().numpy()
            
            # Оригинальная маска (если есть)
            original_mask = item.get('mask')
            
            # Отображение оригинального изображения
            ax = axes[row][0]
            ax.imshow(np.array(image))
            ax.set_title("Оригинальное изображение", fontsize=10)
            ax.axis('off')
            
            # Отображение предсказанной маски
            ax = axes[row][1]
            ax.imshow(predicted_mask, cmap='tab20')
            ax.set_title("Предсказанная маска", fontsize=10)
            ax.axis('off')
            
            # Отображение оригинальной маски
            ax = axes[row][2]
            if original_mask is not None:
                original_mask_resized = original_mask.resize((512, 512))
                ax.imshow(np.array(original_mask_resized), cmap='tab20')
                ax.set_title("Оригинальная маска", fontsize=10)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                ax.set_title("Оригинальная маска", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()