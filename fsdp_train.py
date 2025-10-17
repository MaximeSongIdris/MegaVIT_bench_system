import os
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

import socket
from functools import partial
from hostlist import expand_hostlist
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


# 0. Setup DDP and FSDP for SLURM
def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_rank():
    """Get the rank of the current process"""
    return dist.get_rank()

def get_world_size():
    """Get the total number of processes"""
    return dist.get_world_size()

def is_main_process():
    """Check if this is the main process"""
    return get_rank() == 0

def setup():
    """Initialize distributed training with SLURM"""
    # SLURM environment variables
    rank = int(os.environ.get("SLURM_PROCID", 0))   
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    
    # Set environment variables for PyTorch
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    if "SLURM_JOB_NODELIST" in os.environ:  # Get master address and port from SLURM
        hostnames = expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        master_addr = hostnames[0]
        os.environ["MASTER_ADDR"] = master_addr
    else:
        os.environ["MASTER_ADDR"] = "localhost"
    
    os.environ["MASTER_PORT"] = str(10000 + int(os.environ["SLURM_JOB_ID"]) % 10000)
    
    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    # Summary
    if is_main_process():
        PREFIX = "%i - " % rank
        print(PREFIX + "Number of nodes: %i" % int(os.environ["SLURM_JOB_NUM_NODES"]))
        print(PREFIX + "Node ID        : %i" % int(os.environ["SLURM_NODEID"]))
        print(PREFIX + "World size     : %i" % world_size)
        print(PREFIX + "GPUs per node  : %i" % int(os.environ["SLURM_GPUS_ON_NODE"]))
        print(PREFIX + "Local rank     : %i" % local_rank)
        print(PREFIX + "Master node    : %s" % master_addr)
        print(PREFIX + "Hostname       : %s" % socket.gethostname())
        print(PREFIX + "Port           : %s" % os.environ["MASTER_PORT"])

    return torch.device(f'cuda:{local_rank}')

# 1. Setup and Imports
device = setup()

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

# Use DistributedSampler for multi-GPU training
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=get_world_size(),
    rank=get_rank(),
    shuffle=True
)

val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset,
    num_replicas=get_world_size(),
    rank=get_rank(),
    shuffle=False
)

batch_size = 64
batch_size_per_gpu = batch_size // get_world_size()
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size_per_gpu,
                          sampler=train_sampler,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=False)
val_loader = DataLoader(val_dataset,
                        batch_size=batch_size_per_gpu,
                        sampler=val_sampler,
                        num_workers=4,
                        pin_memory=True,
                        shuffle=False)

# Calculate total steps in one epoch
total_steps = len(train_loader)
if is_main_process():
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
if is_main_process():
    print('Number of params: ', sum(p.numel() for p in model.parameters()))

# Wrap policy for automatic sharding
auto_wrap_policy = partial(
    size_based_auto_wrap_policy,
    min_num_params=1e9  # Wrap modules with at least 1B parameters
)

# Wrap model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    device_id=device,
)

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

    # Only show progress bar on main process
    if is_main_process():
        pbar = tqdm(loader, desc="Training", leave=False)
    else:
        pbar = loader
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
        
        # Update progress bar with current metrics (only on main process)
        if is_main_process():
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / ((pbar.n + 1) * loader.batch_size):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

    # Gather metrics from all processes
    total_loss_tensor = torch.tensor([total_loss], device=device)
    correct_tensor = torch.tensor([correct], device=device)
    
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        
    avg_loss = total_loss_tensor.item() / (total_steps*get_world_size())
    avg_acc = correct_tensor.item() / len(train_dataset)
    
    return avg_loss, avg_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    # Only show progress bar on main process
    if is_main_process():
        pbar = tqdm(loader, desc="Validation", leave=False)
    else:
        pbar = loader

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar (only on main process)
            if is_main_process():
                pbar.set_postfix({
                    'acc': f'{correct / ((pbar.n + 1) * loader.batch_size):.4f}'
                })
    
    # Gather metrics from all processes
    total_loss_tensor = torch.tensor([total_loss], device=device)
    correct_tensor = torch.tensor([correct], device=device)
    
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / (len(loader) * get_world_size())
    avg_acc = correct_tensor.item() / len(val_dataset)
            
    return avg_loss, avg_acc

# Train for 1 epoch
if is_main_process():
    print("\nStarting training for 1 epoch...")
train_loader.sampler.set_epoch(0)

train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
if is_main_process():
    print(f"\nTraining completed - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
if is_main_process():
    print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# 5. Final Steps - Save only on main process
save_policy = StateDictType.FULL_STATE_DICT
with FSDP.state_dict_type(model, save_policy):
    full_state_dict = model.state_dict()
if is_main_process():
    torch.save(full_state_dict, "mega_vit_model.pth")
print("Training finished. Model saved.")

