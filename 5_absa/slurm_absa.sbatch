#!/bin/bash
#SBATCH --job-name=beyond-stars-absa
#SBATCH --output=slurm-absa-%j.out
#SBATCH --error=slurm-absa-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=macs30113

module load python
source ~/beyond-stars-venv/bin/activate

python absa.py \
    --bucket evahu-beyond-stars-raw \
    --local_dir /scratch/midway3/$USER/beyond-stars-absa
