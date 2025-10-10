import os
os.environ['PYTHONPATH'] = '/lustre/work/sos/ssos027/test_multi_noeuds/image/MegaVIT'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from mega_vit.main import MegaVit

# 1. Setup and Imports
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Using CIFAR-10 for demonstration purposes
cifar10 = datasets.CIFAR10(root="./data", download=False, transform=transform)
train_size = int(0.9 * len(cifar10))
val_size = len(cifar10) - train_size
train_dataset, val_dataset = random_split(cifar10, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Calculate total steps in one epoch
total_steps = len(train_loader)
print(f"Total steps in one epoch: {total_steps}")

# 3. Model Initialization
model = MegaVit(
    image_size = 224,
    patch_size = 14,
    num_classes = 10,
    dim = 6144,
    depth = 48,
    heads = 48,
    #mlp_dim = 24576,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)
print('Number of params: ', sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0002)
warmup_steps = int(0.1 * total_steps)

# Warm-up + Cosine schedule for the learning rate
def lr_schedule(step):
    if step < warmup_steps:
        # Warmup phase
        return step / warmup_steps
    else:
        # Cosine annealing phase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))

scheduler = LambdaLR(optimizer, lr_schedule)

# 4. Training Loop
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    
    # Wrap the loader with tqdm for progress bar
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct / ((pbar.n + 1) * loader.batch_size):.4f}'
        })
        
    return total_loss / len(loader), correct / len(train_dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), correct / len(val_dataset)

# Train for 1 epoch
print("\nStarting training for 1 epoch...")
train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
print(f"\nTraining completed - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

print("\nRunning validation...")
val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# 5. Final Steps
torch.save(model.state_dict(), "mega_vit_model.pth")
print("Training finished.")
