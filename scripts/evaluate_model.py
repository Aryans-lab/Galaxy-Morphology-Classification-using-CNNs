import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split  # MISSING IMPORT ADDED
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import BASE_DIR, PROCESSED_DIR, LOG_DIR

# UPDATED CONFIGURATION (100x100 images)
CONFIG = {
    "data_path": os.path.join(PROCESSED_DIR, "galaxy_dataset_balanced_100x100.npz"),  # Updated to 100x100
    "model_path": os.path.join(BASE_DIR, "models", "galaxy_classifier.keras"),
    "val_size": 0.15,
    "batch_size": 32,
    "input_shape": (100, 100, 3),  # Updated to 100x100
    "evaluation_dir": os.path.join(BASE_DIR, "evaluation"),
    "log_file": os.path.join(LOG_DIR, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
}

# Setup logging and directories
os.makedirs(CONFIG['evaluation_dir'], exist_ok=True)
logging.basicConfig(
    filename=CONFIG['log_file'],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("🚀 Starting model evaluation")

def load_data():
    """Load and split dataset identically to training"""
    with np.load(CONFIG['data_path']) as data:
        X, y = data['images'], data['labels']
    
    # Recreate train/test split using same seed as training
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['val_size'], 
        stratify=y, 
        random_state=42
    )
    logging.info(f"📊 Test set size: {len(X_test)} samples")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Perform comprehensive model evaluation"""
    # Basic evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"📈 Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    
    # Classification report
    report = classification_report(
        y_test, 
        y_pred_classes, 
        target_names=['Elliptical', 'Spiral'],
        digits=4
    )
    logging.info(f"Classification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Elliptical', 'Spiral']
    ).plot(cmap='Blues', ax=ax)
    plt.title('Galaxy Classification Confusion Matrix')
    cm_path = os.path.join(CONFIG['evaluation_dir'], "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"💾 Saved confusion matrix to {cm_path}")
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    metrics_path = os.path.join(CONFIG['evaluation_dir'], "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"💾 Saved metrics to {metrics_path}")
    
    # Per-class accuracy
    class_acc = []
    for i, class_name in enumerate(['Elliptical', 'Spiral']):
        acc = cm[i][i] / cm[i].sum()
        logging.info(f"🎯 {class_name} Accuracy: {acc:.2%}")
        class_acc.append(acc)
    
    return metrics

def run_evaluation():
    """Main evaluation workflow"""
    try:
        # Load data and model
        X_test, y_test = load_data()
        model = tf.keras.models.load_model(CONFIG['model_path'])
        logging.info("✅ Model loaded successfully")
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        logging.info("✅ Evaluation completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if run_evaluation():
        print("✅ Evaluation completed. Results saved in", CONFIG['evaluation_dir'])
    else:
        print("❌ Evaluation failed - check logs")