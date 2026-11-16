import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import ISLVideoDataset, create_splits, CLASSES
from src.model import ISLGestureRecognizer

def plot_confusion_matrix(cm, classes, save_path='eval_results/confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix - Indian Sign Language (ISL)',
           ylabel='True Gesture',
           xlabel='Predicted Gesture')

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved confusion matrix plot: {save_path}')


def plot_training_history(history_path='eval_results/history.json', save_path='eval_results/training_curves.png'):
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', markersize=4, label='Val Loss')
    ax1.set_title('Cross-Entropy Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Accuracy (%)')
    ax2.plot(epochs, history['val_acc'], 'r-s', markersize=4, label='Val Accuracy (%)')
    ax2.axhline(y=85.0, color='green', linestyle='--', label='Target Threshold (85%)')
    ax2.set_title('Model Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved training curves plot: {save_path}')


def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/best_isl_model.pth'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize and load weights
    model = ISLGestureRecognizer(
        num_classes=len(CLASSES),
        num_frames=16,
        in_channels=3,
        feature_dim=256,
        hidden_dim=128
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    _, val_samples = create_splits(val_ratio=0.2, seed=42)
    val_dataset = ISLVideoDataset(val_samples, num_frames=16, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for frames, labels, _ in val_loader:
            frames = frames.to(device)
            logits, _ = model(frames)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Classification Report
    report_dict = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True)
    report_str = classification_report(all_targets, all_preds, target_names=CLASSES, digits=4)
    print('\n================ CLASSIFICATION REPORT ================')
    print(report_str)
    
    with open('eval_results/classification_report.txt', 'w') as f:
        f.write('================ ISL RECOGNITION CLASSIFICATION REPORT ================\n')
        f.write(report_str)
        f.write(f'\nTotal Evaluation Samples: {len(all_targets)}\n')
        f.write(f'Overall Accuracy: {report_dict["accuracy"]*100:.2f}%\n')
        
    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, CLASSES, save_path='eval_results/confusion_matrix.png')
    
    # 3. Training curves
    if os.path.exists('eval_results/history.json'):
        plot_training_history('eval_results/history.json', 'eval_results/training_curves.png')
        
    summary = {
        'model': 'Spatio-Temporal CNN + BiGRU + Temporal Attention',
        'classes': CLASSES,
        'val_accuracy_percent': report_dict['accuracy'] * 100.0,
        'macro_avg_f1': report_dict['macro avg']['f1-score'],
        'total_val_samples': len(all_targets),
        'target_accuracy_threshold': 85.0,
        'threshold_satisfied': bool(report_dict['accuracy'] * 100.0 >= 85.0)
    }
    with open('eval_results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f'Overall Validation Accuracy: {report_dict["accuracy"]*100:.2f}% (Target: >=85%)')
    return summary

if __name__ == '__main__':
    evaluate_model()
