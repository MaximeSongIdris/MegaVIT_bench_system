#!/bin/bash

## JOB INFO
#SBATCH --job-name=fsdp_profile_1GPU
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.out

## NODE CONFIGURATION
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --time=02:00:00


## ENV ACTIVATION
export PYTHONPATH=/lustre/work/sos/ssos027/test_multi_noeuds/MegaVIT_bench_system/MegaVIT

## CODE EXECUTION
export NCCL_NET_GDR_LEVEL=LOC
echo $NCCL_NET_GDR_LEVEL
export NCCL_MNNVL_ENABLE=0
echo $NCCL_MNNVL_ENABLE
export NCCL_DEBUG=INFO

time srun singularity exec --nv $WORK/test_multi_noeuds/pytorch-25.08-py3.sif python fsdp_profile.py
