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

unset LD_LIBRARY_PATH

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache
export WANDB_API_KEY=$(cat ~/.wandb_key)
export MUJOCO_GL=egl

cd $SCRATCH/online-q-chunking
cp -rf ~/online-q-chunking/* .

source .venv/bin/activate

env=""
group=""
random_replanning=0
big_net=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      env="$2"
      shift 2
      ;;
    --group)
      group="$2"
      shift 2
      ;;
    --random_replanning)
      random_replanning=1
      shift
      ;;
    --big_net)
      big_net=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$env" ]]; then
    echo "Error: --env is required"
    exit 1
fi

if [[ -z "$group" ]]; then
    echo "Error: --group is required"
    exit 1
fi

FLAGS="--num_evals 64 \
  --batch_size 256 \
  --num_envs 512 \
  --num_eval_envs 512 \
  --discounting 0.99 \
  --action_repeat 1 \
  --unroll_length 60 \
  --min_replay_size 1000 \
  --max_replay_size 10000 \
  --contrastive_loss_fn sym_infonce \
  --energy_fn l2 \
  --train_step_multiplier 1 \
  --log_wandb"

if [[ $random_replanning -eq 1 ]]; then
    FLAGS="$FLAGS --random_replanning"
fi

if [[ $big_net -eq 1 ]]; then
    FLAGS="$FLAGS --total_env_steps 120000000 --n_hidden 6 --use_ln"
else
    FLAGS="$FLAGS --total_env_steps 60000000"
fi

for seed in 0 1 2 3 4; do
    for action_chunk_length in 1 3 5 10 15; do
        for target_entropy_coeff in 0.5 2.0 4.0; do
            jaxgcrl accrl \
                --env "$env" \
                --action_chunk_length $action_chunk_length \
                --target_entropy_coeff $target_entropy_coeff \
                --seed $seed \
                --wandb_group "$group" \
                --exp_name "${env}_acl_${action_chunk_length}_seed_${seed}" \
                $FLAGS
        done
    done
done
