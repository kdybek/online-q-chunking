#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=setup.out
#SBATCH --error=setup.err

ml Python/3.11.5
ml CUDA/12.8.0
ml cuDNN/8.9.7.29-CUDA-12.8.0

export XDG_CACHE_HOME=$SCRATCH/.cache
export CUDA_HOME=$EBROOTCUDA
export CUDNN_HOME=$EBROOTCUDNN

cd $SCRATCH
mkdir -p online-q-chunking
cd online-q-chunking
cp -rf ~/online-q-chunking/* .
python -m venv .venv
source .venv/bin/activate

pip install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
