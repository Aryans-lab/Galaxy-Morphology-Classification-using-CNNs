import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score
from utils import BASE_DIR, LOG_DIR, PROCESSED_DIR

# =================================================================
# PUBLICATION-QUALITY VISUALIZATION SETTINGS
# =================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.bbox': 'tight'
})
sns.set_style("whitegrid", {'grid.linestyle': '--'})

# =================================================================
# CONFIGURATION
# =================================================================
CONFIG = {
    "history_dir": LOG_DIR,
    "model_path": os.path.join(BASE_DIR, "models", "galaxy_classifier.keras"),
    "data_path": os.path.join(PROCESSED_DIR, "galaxy_dataset_balanced_100x100.npz"),
    "evaluation_dir": os.path.join(BASE_DIR, "evaluation"),
    "visualization_dir": os.path.join(BASE_DIR, "visualization", "publication"),
    "input_shape": (100, 100, 3),
    "sample_count": 16,
    "class_names": ["Elliptical", "Spiral"],
    "class_colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
    "pub_dpi": 300
}

# =================================================================
# CORE VISUALIZATION FUNCTIONS
# =================================================================
def create_publication_plots():
    """Generate all visualizations for publication"""
    os.makedirs(CONFIG['visualization_dir'], exist_ok=True)
    print("\n===== GENERATING PUBLICATION VISUALIZATIONS =====")
    
    try:
        # Load essential data
        history = load_training_history()
        model = load_model()
        with np.load(CONFIG['data_path']) as data:
            images, labels = data['images'], data['labels']
        
        # Get test set for evaluation metrics
        _, X_test, _, y_test = train_test_split(
            images, labels, test_size=0.15, stratify=labels, random_state=42
        )
        y_pred = model.predict(X_test, verbose=0)
        y_pred_probs = y_pred.flatten()
        y_pred_classes = (y_pred_probs > 0.5).astype(int)
        
        # Generate core visualizations
        plot_training_history(history)
        plot_class_distribution(labels)
        plot_sample_predictions(model, images, labels)
        plot_confusion_matrix(y_test, y_pred_classes)
        plot_roc_curve(y_test, y_pred_probs)
        plot_precision_recall_curve(y_test, y_pred_probs)
        
        # Additional UG-friendly visualizations
        plot_calibration_curve(y_test, y_pred_probs)
        plot_error_analysis(model, X_test, y_test, y_pred_classes)
        plot_per_class_metrics(y_test, y_pred_classes)
        plot_galaxy_examples(images, labels)
        
        print("\n✅ All publication visualizations generated successfully!")
        print(f"📂 Output directory: {CONFIG['visualization_dir']}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()

# =================================================================
# DATA LOADING AND UTILITIES
# =================================================================
def load_training_history():
    """Load latest training history"""
    history_files = [f for f in os.listdir(CONFIG['history_dir']) 
                   if f.startswith('training_history_') and f.endswith('.json')]
    if not history_files:
        raise FileNotFoundError("No training history files found")
    history_files.sort(reverse=True)
    with open(os.path.join(CONFIG['history_dir'], history_files[0]), 'r') as f:
        return json.load(f)

def load_model():
    """Load model with fallback naming"""
    try:
        return tf.keras.models.load_model(CONFIG['model_path'])
    except:
        alt_path = os.path.join(BASE_DIR, "models", "ML-Glaxay classifier- Model- galaxy_classifier.keras")
        return tf.keras.models.load_model(alt_path)

def train_test_split(X, y, test_size=0.15, stratify=None, random_state=42):
    """Simple train/test split implementation"""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)

# =================================================================
# PLOTTING FUNCTIONS
# =================================================================
def plot_training_history(history):
    """Training history with smoothed curves"""
    # Apply simple smoothing for cleaner plots
    def smooth_curve(points, factor=0.8):
        smoothed = []
        for point in points:
            if smoothed:
                prev = smoothed[-1]
                smoothed.append(prev * factor + point * (1 - factor))
            else:
                smoothed.append(point)
        return smoothed
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(smooth_curve(history['accuracy']), label='Training', linewidth=2)
    plt.plot(smooth_curve(history['val_accuracy']), label='Validation', linewidth=2)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(smooth_curve(history['loss']), label='Training', linewidth=2)
    plt.plot(smooth_curve(history['val_loss']), label='Validation', linewidth=2)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'training_history.png'), 
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_class_distribution(labels):
    """Class distribution with annotations"""
    class_counts = [sum(labels == 0), sum(labels == 1)]
    total = sum(class_counts)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(CONFIG['class_names'], class_counts, 
                   color=CONFIG['class_colors'], alpha=0.85)
    
    plt.title('Galaxy Class Distribution')
    plt.ylabel('Number of Galaxies')
    
    # Annotate bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}\n({count/total:.1%})', 
                 ha='center', va='bottom', fontsize=12)
    
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'class_distribution.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_sample_predictions(model, images, labels):
    """Sample predictions with confidence scores"""
    indices = np.random.choice(len(images), CONFIG['sample_count'], replace=False)
    sample_images = images[indices]
    sample_labels = labels[indices]
    predictions = model.predict(sample_images, verbose=0)
    confidences = predictions.flatten()
    pred_classes = (predictions > 0.5).astype(int).flatten()
    
    plt.figure(figsize=(12, 12))
    for i in range(CONFIG['sample_count']):
        plt.subplot(4, 4, i+1)
        plt.imshow(sample_images[i].astype('uint8'))
        
        actual = CONFIG['class_names'][sample_labels[i]]
        predicted = CONFIG['class_names'][pred_classes[i]]
        confidence = confidences[i]
        color = 'green' if sample_labels[i] == pred_classes[i] else 'red'
        
        plt.title(f"Actual: {actual}\nPred: {predicted}\nConf: {confidence:.3f}", 
                  color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Galaxy Classification Examples', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'sample_predictions.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Confusion matrix with annotations"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CONFIG['class_names'], 
                yticklabels=CONFIG['class_names'],
                annot_kws={"fontsize":12})
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'confusion_matrix.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_roc_curve(y_true, y_scores):
    """ROC curve with AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=CONFIG['class_colors'][1], lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'roc_curve.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_precision_recall_curve(y_true, y_scores):
    """Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color=CONFIG['class_colors'][1], lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'precision_recall_curve.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_calibration_curve(y_true, y_scores):
    """Calibration curve for probability reliability"""
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'calibration_curve.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_error_analysis(model, X_test, y_test, y_pred):
    """Visualize misclassified examples"""
    # Identify misclassified samples
    incorrect = np.where(y_pred != y_test)[0]
    if len(incorrect) == 0:
        print("⚠️ No misclassified samples found")
        return
        
    # Select up to 8 examples
    sample_indices = incorrect[:min(8, len(incorrect))]
    sample_images = X_test[sample_indices]
    sample_true = y_test[sample_indices]
    predictions = model.predict(sample_images, verbose=0)
    confidences = predictions.flatten()
    pred_classes = (predictions > 0.5).astype(int).flatten()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 4, i+1)
        plt.imshow(sample_images[i].astype('uint8'))
        
        actual = CONFIG['class_names'][sample_true[i]]
        predicted = CONFIG['class_names'][pred_classes[i]]
        confidence = confidences[i]
        
        plt.title(f"Actual: {actual}\nPred: {predicted}\nConf: {confidence:.3f}", 
                  color='red', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Misclassified Galaxy Examples', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'misclassified_examples.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_per_class_metrics(y_true, y_pred):
    """Bar chart of precision, recall, F1 per class"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = {
        'Precision': precision_score(y_true, y_pred, average=None),
        'Recall': recall_score(y_true, y_pred, average=None),
        'F1-score': f1_score(y_true, y_pred, average=None)
    }
    
    x = np.arange(len(CONFIG['class_names']))  # label locations
    width = 0.25  # bar width
    multiplier = 0
    
    plt.figure(figsize=(10, 6))
    
    for metric, values in metrics.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, values, width, label=metric,
                        color=CONFIG['class_colors'][multiplier % len(CONFIG['class_colors'])])
        plt.bar_label(rects, padding=3, fmt='%.3f')
        multiplier += 1
    
    plt.ylabel('Score')
    plt.title('Per-class Performance Metrics')
    plt.xticks(x + width, CONFIG['class_names'])
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'per_class_metrics.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

def plot_galaxy_examples(images, labels):
    """Show representative examples of each class"""
    plt.figure(figsize=(10, 5))
    
    # Elliptical examples
    elliptical_idx = np.where(labels == 0)[0][:4]
    for i, idx in enumerate(elliptical_idx):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[idx].astype('uint8'))
        plt.title('Elliptical', fontsize=10)
        plt.axis('off')
    
    # Spiral examples
    spiral_idx = np.where(labels == 1)[0][:4]
    for i, idx in enumerate(spiral_idx):
        plt.subplot(2, 4, i+5)
        plt.imshow(images[idx].astype('uint8'))
        plt.title('Spiral', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Galaxy Type Examples', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['visualization_dir'], 'galaxy_examples.png'),
                dpi=CONFIG['pub_dpi'])
    plt.close()

# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    create_publication_plots()