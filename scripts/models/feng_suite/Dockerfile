# CPU environment reproducing Feng et al. 2024 SL_benchmark (py3.7, TF1.15, torch 1.5, dgl 0.4.3, PyG 1.5)
FROM python:3.7-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.18.5 scipy==1.7.3 pandas==1.3.5 scikit-learn==1.0.2 networkx==2.6.3 \
    tensorflow==1.15.5 protobuf==3.19.0 h5py==2.10.0 tqdm lmdb==0.98 pyarrow==12.0.1 wandb==0.15.8 matplotlib==3.5.3 \
    && pip install --no-cache-dir torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir torch-scatter==2.0.4 torch-sparse==0.6.4 torch-cluster==1.5.4 torch-spline-conv==1.2.0 -f https://data.pyg.org/whl/torch-1.5.0+cpu.html \
    && pip install --no-cache-dir torch-geometric==1.5.0 dgl==0.4.3post2
ENV WANDB_MODE=disabled
WORKDIR /work
