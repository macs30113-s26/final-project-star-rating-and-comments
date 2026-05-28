#!/bin/bash
#SBATCH --job-name=beyond-stars-embed
#SBATCH --output=slurm-embed-%j.out
#SBATCH --error=slurm-embed-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=macs30113

module load python cuda
source ~/beyond-stars-venv/bin/activate

python embed_reviews.py \
    --bucket evahu-beyond-stars-raw \
    --local_dir /scratch/midway3/$USER/beyond-stars-embed
