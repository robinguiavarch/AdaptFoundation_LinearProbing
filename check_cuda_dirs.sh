#!/bin/bash
#SBATCH --job-name=check_cuda_dirs
#SBATCH --output=logs/check_cuda_dirs_%j.out
#SBATCH --error=logs/check_cuda_dirs_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=2G

echo "==========================="
echo "LIST /usr/local/cuda*"
echo "==========================="

ls -ld /usr/local/cuda*
