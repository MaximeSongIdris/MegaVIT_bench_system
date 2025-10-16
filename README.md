## Clone repository and its submodules
```sh
git clone --recursive https://github.com/MaximeSongIdris/MegaVIT_bench_system.git
```

## Download CIFAR10
```python
from torchvision import datasets
datasets.CIFAR10(root="./data", download=True)
```

## Patch MegaVIT submodule
MegaVIT has a dependency with Poetry, which we want to remove when installing on site-packages.
```sh
mv MegaVIT/pyproject.toml MegaVIT/pyproject_old.toml
cp pyproject_MegaVIT.toml MegaVIT/pyproject.toml
```

## Benchmark Results

On DALIA, we are using a singularity container. RDMA deactivated...

| Model                                       | Platform                                  | Configuration | Time   |
|---------------------------------------------|-------------------------------------------|---------------|--------|
| VisionTransformer 4.8B sur CIFAR10 (50 000) | DALIA (2 noeuds / 1 GPU par noeud / FSDP) | Sans RDMA     | 9m40s  |
|                                             |                                           | Défaut        | 9m39s  |
|                                             |                                           | Force RDMA    | 9m41s  |
|                                             | H100 (2 noeuds / 2 GPU par noeud / FSDP)  | Sans RDMA     | 51m41s |
|                                             |                                           | Défaut        | 43m29s |
