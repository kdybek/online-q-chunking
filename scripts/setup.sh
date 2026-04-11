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
cuDNN/8.9.7.29-CUDA-12.8.0

export XDG_CACHE_HOME=$SCRATCH/.cache
export CUDA_HOME=$EBROOTCUDA
export CUDNN_HOME=$EBROOTCUDNN

cd $SCRATCH
mkdir -p online-q-chunking
cd online-q-chunking
cp -rf ~/online-q-chunking/* .
python -m venv .venv
source .venv/bin/activate

# wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O bazel
# chmod +x bazel
# export PATH="$PWD:$PATH"
# export USE_BAZEL_VERSION=6.1.2

wget https://github.com/jax-ml/jax/archive/refs/tags/jaxlib-v0.4.25.zip
unzip jaxlib-v0.4.25.zip
cd jax-jaxlib-v0.4.25

python build/build.py \
  --enable_cuda \
  --cuda_path=$CUDA_HOME \
  --cudnn_path=$CUDNN_HOME \
  --cuda_version=12 \
  --cudnn_version=8

pip install dist/jaxlib-*.whl

pip install -e .
