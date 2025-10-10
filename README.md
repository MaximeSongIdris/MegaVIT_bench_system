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
