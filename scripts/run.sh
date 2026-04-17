#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

FLAGS="--num_evals 200 \
  --total_env_steps 60000000 \
  --batch_size 256 \
  --num_envs 512 \
  --discounting 0.99 \
  --action_repeat 1 \
  --unroll_length 62 \
  --min_replay_size 1000 \
  --max_replay_size 10000 \
  --contrastive_loss_fn sym_infonce \
  --energy_fn l2 \
  --train_step_multiplier 1 \
  --log_wandb"

ENV=$1

for seed in 0 1 2; do
    for action_chunk_length in 1 3 5; do
        jaxgcrl accrl \
            --env $ENV \
            --action_chunk_length $action_chunk_length \
            --replan_every $action_chunk_length \
            --seed $seed \
            --wandb_group "online_q_chunking" \
            --exp_name "${ENV}_acl_${action_chunk_length}_seed_${seed}" \
            $FLAGS
    done
done

rm -rf $VENV
