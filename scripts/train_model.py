import os
import json
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, models, callbacks, Input
from sklearn.model_selection import train_test_split
from utils import BASE_DIR, PROCESSED_DIR, LOG_DIR

# Timestamp for unique file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configuration
CONFIG = {
    "data_path": os.path.join(PROCESSED_DIR, "galaxy_dataset_balanced_100x100.npz"),
    "val_size": 0.15,
    "batch_size": 32,
    "epochs": 20,
    "patience": 5,
    "input_shape": (100, 100, 3),
    "model_save_path": os.path.join(BASE_DIR, "models", "galaxy_classifier.keras"),
    "history_save_path": os.path.join(LOG_DIR, f"training_history_{timestamp}.json"),
    "tensorboard_log_dir": os.path.join(LOG_DIR, "tensorboard", timestamp),
    "log_file": os.path.join(LOG_DIR, f"train_{timestamp}.log")
}

# Logging setup
logging.basicConfig(
    filename=CONFIG['log_file'],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("🚀 Training started")

def load_data():
    """Load and split dataset"""
    with np.load(CONFIG['data_path']) as data:
        X, y = data['images'], data['labels']
    logging.info(f"📊 Loaded data with shape {X.shape}")
    return train_test_split(X, y, test_size=CONFIG['val_size'], stratify=y, random_state=42)

def build_model():
    """Fixed CNN model matching your data shape"""
    model = models.Sequential([
        Input(shape=CONFIG['input_shape']),
        layers.Rescaling(1/255.0),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.summary(print_fn=lambda x: logging.info(x))
    return model

def save_history(history):
    """Save training history to JSON"""
    with open(CONFIG['history_save_path'], 'w') as f:
        json.dump(history.history, f, indent=4)
    logging.info(f"📈 Saved training history to {CONFIG['history_save_path']}")

def train():
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
    os.makedirs(CONFIG['tensorboard_log_dir'], exist_ok=True)

    X_train, X_val, y_train, y_val = load_data()

    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[
            callbacks.TensorBoard(log_dir=CONFIG['tensorboard_log_dir'], histogram_freq=1),
            callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['patience'], restore_best_weights=True),
            callbacks.ModelCheckpoint(CONFIG['model_save_path'], save_best_only=True)
        ],
        verbose=1
    )

    save_history(history)
    logging.info(f"✅ Training complete. Model saved at {CONFIG['model_save_path']}")
    logging.info(f"🗂️ TensorBoard logs at {CONFIG['tensorboard_log_dir']}")

if __name__ == "__main__":
    train()
