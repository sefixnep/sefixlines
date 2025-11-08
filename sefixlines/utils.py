import os
import torch
import random
import numpy as np
import torch.nn as nn


def set_all_seeds(seed=42):
    # Устанавливаем seed для хэш-функции Python (опция для контроля поведения хэшей)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Устанавливаем seed для встроенного генератора Python
    random.seed(seed)
    # Устанавливаем seed для NumPy
    np.random.seed(seed)

    # Устанавливаем seed для PyTorch
    torch.manual_seed(seed)
    # Устанавливаем seed для генератора на CUDA
    torch.cuda.manual_seed(seed)

    # Отключаем недетерминированное поведение в алгоритмах CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomOutput(nn.Module):
    def __init__(self, model, output_transform=lambda out: out.logits):
        super().__init__()
        self.model = model
        self.output_transform = output_transform

    def forward(self, *args, **kwargs):
        return self.output_transform(self.model(*args, **kwargs))

    def __getattr__(self, name):
        if name in ('model', 'output_transform'):
            return super().__getattr__(name)
        return getattr(self.model, name)
    
    def __setattr__(self, name, value):
        if name in ('model', 'output_transform'):
            super().__setattr__(name, value)
        else:
            setattr(self.model, name, value)
