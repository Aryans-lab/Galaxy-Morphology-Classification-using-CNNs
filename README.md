# 🌌 Galaxy Morphology Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A deep learning pipeline for automating the morphological classification of galaxies (Spiral vs. Elliptical) using Convolutional Neural Networks (CNNs).**

---

## 📖 Abstract

Galaxy morphology encodes key information about galaxy formation and evolutionary history. Automating morphological classification is essential for modern and upcoming large sky surveys, where manual labeling is infeasible due to the sheer volume of data.

This repository provides a complete, research-grade pipeline designed to classify galaxies from the **Galaxy Zoo 2** dataset into **Spiral** and **Elliptical** classes. The project encompasses the entire lifecycle of a deep learning task: data ingestion, preprocessing, dataset balancing, model training, rigorous evaluation, and publication-ready visualization.

---

## 🏗️ Architectural Overview

The project is structured as a modular pipeline:

1. **Data Ingestion & Filtering:** Raw tabular data from Galaxy Zoo 2 is merged and filtered to select high-confidence morphological labels (confidence > 0.8).
2. **Image Processing:** Galaxy images are mapped to their corresponding labels, resized, and converted into structured numpy arrays.
3. **Dataset Balancing:** To prevent model bias toward the majority class, the dataset is balanced using Random Over-Sampling in a memory-efficient manner.
4. **Model Training:** A custom Convolutional Neural Network (CNN) is trained to perform binary classification. The model utilizes Early Stopping and Model Checkpointing to ensure optimal generalization.
5. **Evaluation & Visualization:** The trained model is evaluated on a held-out test set. Comprehensive metrics (Accuracy, ROC-AUC, PR-AUC) and visualizations are generated to interpret the model's performance and failure modes.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. The required dependencies are listed in `scripts/requirements.txt`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/galaxy-classification.git
   cd galaxy-classification
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r scripts/requirements.txt
   ```

### Data Setup

1. Place the Galaxy Zoo 2 raw data files (`gz2_filename_mapping.csv`, `gz2_hart16.csv`) in `data/raw/`.
2. Extract the raw galaxy images into `data/processed/images/`.

### Execution Order

Run the scripts sequentially from the repository root:

```bash
# 1. Merge and filter raw tabular data
python scripts/merge_and_filter.py

# 2. Process images into numpy arrays
python scripts/process_images.py

# 3. Balance the dataset
python scripts/balance_dataset.py

# 4. Train the model
python scripts/train_model.py

# 5. Evaluate the trained model
python scripts/evaluate_model.py

# 6. Generate visualizations
python scripts/visualise_result.py
```

---

## 🧠 How It Works (Script Breakdown)

The `scripts/` directory houses the core logic, decoupled into single-responsibility modules:

### 1. `merge_and_filter.py`
Merges the `gz2_hart16.csv` debiased morphological vote fractions with `gz2_filename_mapping.csv`. It filters for high-confidence samples where the probability of being either purely smooth (Elliptical) or having distinct features/disk (Spiral) exceeds `0.8`. Outputs `filtered_labels.csv`.

### 2. `process_images.py`
Iterates over the filtered dataset, loading the corresponding JPEG images. Images are standardized to `128x128` pixels and converted into a compressed Numpy archive (`galaxy_dataset.npz`), mapping each image matrix to its binary label.

### 3. `balance_dataset.py`
Addresses class imbalance to ensure the model doesn't blindly predict the majority class. Images are downscaled to `100x100` pixels (for memory efficiency), and the dataset is balanced using **RandomOverSampler**. Outputs the final analysis-ready `galaxy_dataset_balanced_100x100.npz`.

### 4. `train_model.py`
Constructs and trains a tailored VGG-style CNN:
- **Architecture:** 3 Convolutional blocks (Conv2D -> ReLU -> MaxPooling) followed by Dense layers.
- **Optimization:** Adam optimizer with Binary Cross-Entropy loss.
- **Callbacks:** Integrates Early Stopping (with `patience=5`), Model Checkpoints, and TensorBoard logging.

### 5. `evaluate_model.py`
Loads the trained `.keras` model and evaluates it against a stratified hold-out test set (15% of the data). Computes accuracy, loss, generates a classification report, and plots a confusion matrix.

### 6. `visualise_result.py`
Generates publication-ready visualizations stored in `results/figures/`. Functions include plotting training histories, ROC and Precision-Recall curves, probability calibration curves, and visualizing misclassified instances to physically interpret model failures.

### `utils.py`
A central utility module defining project directory constants (`BASE_DIR`, `LOG_DIR`, `PROCESSED_DIR`) and environment setup logic.

---

## 📊 Results & Performance

*Note: Visualizations are generated dynamically by `visualise_result.py`.*

- **Overall Classification Performance:**
  The custom CNN demonstrates robust discriminative capability between elliptical and spiral morphologies, achieving **high accuracy** with a well-calibrated confidence score.
- **Per-Class Metrics:**
  - **Ellipticals (Smooth):** Exhibits high recall. The network easily identifies smooth light profiles and centralized concentrations.
  - **Spirals (Features/Disk):** Exhibits high precision. The network effectively captures edge-on features and spiral arms.
- **Interpretability:**
  Analysis of misclassifications reveals that errors generally stem from physically ambiguous sources, such as low surface-brightness arms, edge-on spirals resembling ellipticals, or foreground star contamination.

---

## 📁 Directory Structure

```text
├── data/
│   ├── raw/                # Raw CSV metadata from Galaxy Zoo
│   └── processed/          # Processed labels, images, and .npz arrays
├── evaluation/             # Metrics and confusion matrices output
├── logs/                   # Training logs, histories, and TensorBoard data
├── models/                 # Saved trained models (.keras)
├── results/
│   └── figures/            # Publication-ready plots from visualise_result.py
├── scripts/
│   ├── balance_dataset.py
│   ├── evaluate_model.py
│   ├── merge_and_filter.py
│   ├── process_images.py
│   ├── train_model.py
│   ├── visualise_result.py
│   ├── utils.py
│   └── requirements.txt
├── README.md               # Project documentation
└── LICENSE                 # Open-source license
```

---

## 🔮 Future Work

- **Multi-Class Extension:** Expanding the taxonomy to include mergers, irregulars, and detailed sub-classifications (e.g., barred vs. unbarred spirals).
- **Transfer Learning:** Integrating modern pre-trained backbones like ResNet50 or EfficientNet to potentially boost feature extraction on lower-resolution samples.
- **Uncertainty Quantification:** Implementing Bayesian Neural Networks or Monte Carlo Dropout to provide probabilistic error bounds on classifications.

---

## 👨‍🔬 Author

**Aryan Bandyopadhyay**  
Integrated MSc Physics  
School of Physical Sciences  
NISER Bhubaneswar
