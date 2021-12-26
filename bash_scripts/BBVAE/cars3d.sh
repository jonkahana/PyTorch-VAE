#!/bin/bash

source /cs/labs/yedid/jonkahana/lord_cl_venv/bin/activate.csh
export CUDA_HOME="/usr/local/nvidia/cuda/11.0/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"

cd /cs/labs/yedid/jonkahana/external/VAEs/PyTorch-VAE

python -u run.py -c configs/BVAE/bbvae__cars3d.yaml