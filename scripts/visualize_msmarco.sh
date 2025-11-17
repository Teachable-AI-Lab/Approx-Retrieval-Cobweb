#!/bin/bash
#SBATCH --job-name=visualize_ms_marco
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron
#SBATCH --output=slurm_outputs/visualize_msmarco.out
#SBATCH --error=slurm_errors/visualize_msmarco.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd ~/flash/Approx-Cobweb
export PYTHONPATH=$(pwd)

# Default arguments - can be overridden by command line arguments
CONFIG_FILE=${1:-"configs/visualize.json"}

echo "Starting MS Marco Visualize benchmark at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/visualize_msmarco.py --config "$CONFIG_FILE"

echo "Visualize MS Marco script completed at $(date)"