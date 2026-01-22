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

import socket
from functools import partial
from hostlist import expand_hostlist
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
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
    """Initialize distributed training.

    With SLURM:
        All environment variables (SLURM_JOB_NODELIST, SLURM_JOB_ID, SLURM_NTASKS, etc.)
        are automatically set by the scheduler. The master port is derived from SLURM_JOB_ID
        to minimize the chance of using an already allocated port.

    Without SLURM (we suppose that it is a mono-gpu script):
        Defaults to localhost with a random port. For multiple single-GPU scripts on the
        same node, set job to select the GPU for each script:
            CUDA_VISIBLE_DEVICES=0 python FSDP_sAC.py

    Returns:
        torch.device: The CUDA device assigned to this process.
    """
    # SLURM environment variables
    master_addr = expand_hostlist(str(os.environ.get("SLURM_JOB_NODELIST", "localhost")))[0]
    master_port = 10000 + int(os.environ.get("SLURM_JOB_ID", random.randint(0, 9999))) % 10000
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    # Set environment variables for PyTorch
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    # Summary
    if is_main_process():
        PREFIX = "%i - " % rank
        print(PREFIX + "Master node    : %s" % master_addr)
        print(PREFIX + "Port           : %s" % master_port)
        print(PREFIX + "World size     : %i" % world_size)
        print(PREFIX + "Number of nodes: %i" % n_nodes)
        print(PREFIX + "GPUs per node  : %i" % gpus_per_node)
        print(PREFIX + "Hostname       : %s" % socket.gethostname())
        print(PREFIX + "Node ID        : %i" % node_id)
        print(PREFIX + "Local rank     : %i" % local_rank)

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
if is_main_process():
    print(len(train_loader))
