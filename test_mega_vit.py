import os
os.environ['PYTHONPATH'] = '/lustre/work/sos/ssos027/test_multi_noeuds/image/MegaVIT'

import torch
from mega_vit.main import MegaVit


# Allocated memory (used by tensors)
allocated = torch.cuda.memory.max_memory_allocated() / (1024**3)
print(f"  Allocated: {allocated:.2f} GB")

v = MegaVit(
    image_size = 224,
    patch_size = 14,
    num_classes = 1000,
    dim = 6144,
    depth = 48,
    heads = 48,
    mlp_dim = 24576,
    dropout = 0.1,
    emb_dropout = 0.1
).to('cuda')

print(sum(p.numel() for p in v.parameters()))

# Allocated memory (used by tensors)
allocated = torch.cuda.memory.max_memory_allocated() / (1024**3)
print(f"  Allocated: {allocated:.2f} GB")

img = torch.randn(1, 3, 224, 224).to('cuda')
preds = v(img) # (1, 1000)
print(preds)

# Allocated memory (used by tensors)
allocated = torch.cuda.memory.max_memory_allocated() / (1024**3)
print(f"  Allocated: {allocated:.2f} GB")
