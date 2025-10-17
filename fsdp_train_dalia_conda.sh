#!/bin/bash

## JOB INFO
#SBATCH --job-name=fsdp_train_2GPUs
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.out

## NODE CONFIGURATION
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --time=02:00:00


## ENV ACTIVATION

# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lustre/work/sos/ssos027/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lustre/work/sos/ssos027/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/lustre/work/sos/ssos027/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/lustre/work/sos/ssos027/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate base
export PYTHONPATH=/lustre/work/sos/ssos027/test_multi_noeuds/MegaVIT_bench_system/MegaVIT

## CODE EXECUTION
export NCCL_NET_GDR_LEVEL=LOC
echo $NCCL_NET_GDR_LEVEL
time srun python fsdp_train.py
