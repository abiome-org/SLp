#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
if ! conda env list | grep -qE '^slbench\s'; then
  conda create -y -n slbench python=3.11 pip
fi
conda activate slbench
python -m pip install -U pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install tensorflow wandb scikit-learn scipy pandas numpy tqdm networkx lmdb h5py
python -m pip install torch-geometric
python -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
python - <<'PY'
import torch
import tensorflow as tf
import dgl
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("tf", tf.__version__)
print("dgl", dgl.__version__)
PY
