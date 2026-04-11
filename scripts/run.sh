#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=baseline.out
#SBATCH --error=baseline.err

module load CUDA/12.9.1

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

cd $SCRATCH/online-q-chunking
cp -rf ~/online-q-chunking/* .
source .venv/bin/activate

jaxgcrl crl --env ant
