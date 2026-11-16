import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import ISLVideoDataset, create_splits, CLASSES
from src.model import ISLGestureRecognizer

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (frames, labels, _) in enumerate(dataloader):
        frames = frames.to(device) # (B, T, C, H, W)
        labels = labels.to(device) # (B,)
        
        optimizer.zero_grad()
        logits, _ = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping to stabilize recurrent layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += loss.item() * frames.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels, _ in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            logits, _ = model(frames)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * frames.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return val_loss, val_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    
    # 1. Prepare datasets
    train_samples, val_samples = create_splits(val_ratio=0.2, seed=42)
    train_dataset = ISLVideoDataset(train_samples, num_frames=16, is_train=True)
    val_dataset = ISLVideoDataset(val_samples, num_frames=16, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 2. Initialize Model
    model = ISLGestureRecognizer(
        num_classes=len(CLASSES),
        num_frames=16,
        in_channels=3,
        feature_dim=256,
        hidden_dim=128
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model initialized. Trainable parameters: {total_params:,}')
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('eval_results', exist_ok=True)
    
    print('Starting training loop...')
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - t0
        print(f'Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join('models', 'best_isl_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': CLASSES,
                'num_frames': 16,
                'feature_dim': 256,
                'hidden_dim': 128
            }, save_path)
            print(f'  --> Best model saved! (Val Acc: {val_acc:.1f}%)')
            
    with open('eval_results/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training complete. Best Validation Accuracy: {best_val_acc:.1f}%')

if __name__ == '__main__':
    main()
