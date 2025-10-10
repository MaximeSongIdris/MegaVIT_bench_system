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
#SBATCH --time=00:30:00

## CODE EXECUTION
time srun singularity exec --nv $PROJECT/test_multi_noeuds/pytorch-25.08-py3.sif python fsdp_train.py
