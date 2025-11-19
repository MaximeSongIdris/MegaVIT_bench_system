#!/bin/bash

## JOB INFO
#SBATCH --job-name=fsdp_train_2Nodes
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.out

## NODE CONFIGURATION
#SBATCH --constraint=h100
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --account=sos@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00

## ENV ACTIVATION
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
export PYTHONPATH=/lustre/fswork/projects/idris/sos/ssos027/bench/MegaVIT_bench_system/MegaVIT

## CODE EXECUTION
export NCCL_NET_GDR_LEVEL=LOC
echo $NCCL_NET_GDR_LEVEL
export NCCL_DEBUG=INFO

time srun python fsdp_train.py
