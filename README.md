# Practical Lab 3 — Vanilla CNN and Fine-Tune VGG16

Author: Maduka Albright Ifechukwude (Student ID: 9053136)

Overview
--------
This project implements and compares two convolutional neural network approaches for the Dogs vs Cats classification task using a small dataset (~5,000 images). The notebook covers data loading and exploratory data analysis (EDA), data augmentation, training a vanilla CNN from scratch, transfer learning with pre-trained VGG16 (feature extraction + fine-tuning), model checkpointing, evaluation, and error analysis.

Repository structure
--------------------
- `Albright_9053136_Practical_Lab3_Vanilla_CNN_and_Fine_Tune.ipynb` — main notebook containing code, plots, and narrative.
- `data/` — dataset root; expected layout:
	- `data/kaggle_dogs_vs_cats_small/train/{cat,dog}`
	- `data/kaggle_dogs_vs_cats_small/validation/{cat,dog}`
	- `data/kaggle_dogs_vs_cats_small/test/{cat,dog}`
- `models_lab3/` — directory for checkpoints and saved weights.

Key results (reported in the notebook)
-------------------------------------
- Vanilla CNN test accuracy: ~0.746
- VGG16 fine-tuned test accuracy: ~0.9725

Environment & Dependencies
--------------------------
Recommended Python environment (example):

- Python 3.10+
- TensorFlow 2.15
- numpy
- matplotlib
- scikit-learn
- jupyter

Install example (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow==2.15 numpy matplotlib scikit-learn jupyter
```

If you prefer, I can generate a `requirements.txt` file that pins exact versions from your environment.

Notebook walkthrough (sections)
--------------------------------
1. Imports, configuration, and random seeds for reproducibility.
2. Data loading using `tf.keras.utils.image_dataset_from_directory` for `train`, `validation`, and `test` splits.
3. Exploratory Data Analysis: class counts, sample images, pixel histograms, and shape checks.
4. Data augmentation pipeline implemented with Keras layers (`RandomFlip`, `RandomRotation`, `RandomZoom`).
5. Vanilla CNN model: architecture, compile, training, and history plotting.
6. VGG16 transfer learning: build base model (`include_top=False`), add classifier head, feature-extraction training, then fine-tune top layers.
7. Callbacks: ModelCheckpoint (save best), EarlyStopping (restore best weights).
8. Evaluation: load best models, compute accuracy, confusion matrix, classification report (precision/recall/F1), precision–recall curve and Average Precision (AP).
9. Error analysis: visualize misclassified test images and discuss failure modes.

Models & training details
-------------------------
- Vanilla CNN: 3 conv blocks (32→64→128 filters), max pooling, flatten, Dense(128), Dropout(0.5), Dense(1, sigmoid). Optimizer: Adam (1e-4); loss: binary_crossentropy.
- VGG16-based: `VGG16(include_top=False, weights='imagenet')` as base, GlobalAveragePooling2D, Dense(256)+Dropout(0.5), Dense(1,sigmoid) head. Feature-extraction uses Adam(1e-4); fine-tuning uses Adam(1e-5) after unfreezing top layers.
