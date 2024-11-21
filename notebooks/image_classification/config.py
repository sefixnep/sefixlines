from torch import cuda

# Устройство (cuda или cpu)
device, task_type = ("cuda", "GPU") if cuda.is_available() else ("cpu", "CPU")

# Параметры для изображения
image_size = (224, 224)
mean, std = 0.5, 0.5