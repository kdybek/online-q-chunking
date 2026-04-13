#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=baseline.out
#SBATCH --error=baseline.err

ml Python/3.11.5
ml CUDA/12.8.0
ml cuDNN/8.9.7.29-CUDA-12.8.0

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

cd $SCRATCH/online-q-chunking
cp -rf ~/online-q-chunking/* .

VENV=".venv_$SLURM_JOB_ID"
python -m venv $VENV
source $VENV/bin/activate

pip install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

jaxgcrl crl --env ant --action_chunk_length 1 &
wait

rm -rf $VENV
