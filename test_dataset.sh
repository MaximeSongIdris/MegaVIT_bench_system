#!/bin/bash

## JOB INFO
#SBATCH --job-name=test_dataset
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --error=slurm_log/%x_%j.out

## NODE CONFIGURATION
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --time=00:30:00

## CODE EXECUTION
time srun singularity exec --nv $WORK/test_multi_noeuds/pytorch-25.08-py3.sif python test_dataset.py
