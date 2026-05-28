#!/bin/bash
#SBATCH --job-name=beyond-stars-topics
#SBATCH --output=slurm-topics-%j.out
#SBATCH --error=slurm-topics-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=macs30113

module load python
source ~/beyond-stars-venv/bin/activate

python discover_topics.py \
    --bucket evahu-beyond-stars-raw \
    --local_dir /scratch/midway3/$USER/beyond-stars-topics
