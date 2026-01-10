<div align="center">

# 🚀 Sefixlines Model Pipelines

<img src="https://raw.githubusercontent.com/sefixnep/sefixlines/main/assets/logo.png" alt="Sefixlines Logo" width="500"/>

<br>

[![PyPI](https://img.shields.io/badge/PyPI-Install%20Package-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/sefixlines/) &nbsp; [![GitHub](https://img.shields.io/badge/GitHub-View%20Source-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sefixnep/sefixlines)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/sefixnep) &nbsp; [![GitHub Profile](https://img.shields.io/badge/GitHub%20Profile-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sefixnep)

</div>

> 🆕 **UPDATE**: Regression tasks   
> 🆕 **UPDATE**: Text classification  

## ✨ Features
- ⚡ Quick start without tons of code
- 🖼️ Image and text classification
- 🎯 Image semantic segmentation
- 💾 Automatic weights saving/loading
- 🔧 Easy customization (loss_fn, optimizer, scheduler, augmentation)

## ⚙️ Installation
```bash
pip install sefixlines
```

## 🎯 Get Started with Basic Example

For a quick start, use ready-made templates with configured pipelines:

```python
from sefixlines import baseline

# Creates a ready-to-use notebook with an example for your task
baseline.create('raw')                         # Universal
baseline.create('image_classification')        # Image classification
baseline.create('text_classification')         # Text classification
baseline.create('image_regression')            # Image regression
baseline.create('text_regression')             # Text regression
baseline.create('image_semantic_segmentation') # Semantic segmentation
```

This command will create a `sefixline.ipynb` file in the current directory with a fully working example, including:
- 📊 Data loading and preparation
- 🤖 Model setup
- 🏋️ Training with visualization
- 📈 Results evaluation

> ⚡ **This is the fastest way to get started!** Just open the created notebook and adapt it to your data.

## 🚦 Minimal Manual Run
1. **Prepare your data**
```python
from sefixlines import datasets

datasets.ImageClassificationDataset(paths, labels)                  # Image classification
datasets.TextClassificationDataset(texts, labels)                   # Text classification
datasets.ImageRegressionDataset(paths, labels)                      # Image regression
datasets.TextRegressionDataset(texts, labels)                       # Text regression
datasets.ImageSemanticSegmentationDataset(image_paths, mask_paths)  # Semantic segmentation
```
2. **Choose a model** (any model that returns logits).
3. **Train**
```python
from sefixlines import models

# For classification
model_wrapper = models.Classifier(model, "MyModel")
model_wrapper.fit(train_set, valid_set, num_epochs=3)

# For regression
regressor = models.Regressor(model, "MyRegressor")
regressor.fit(train_set, valid_set, num_epochs=3)

# For semantic segmentation
segmenter = models.SemanticSegmenter(model, "MySemanticSegmenter")
segmenter.fit(train_set, valid_set, num_epochs=3)
```

License
-------

MIT. See LICENSE file.
