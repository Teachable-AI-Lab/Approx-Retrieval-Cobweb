#!/bin/bash
#SBATCH --job-name=benchmark_ms_marco
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --output=slurm_outputs/ms_marco.out
#SBATCH --error=slurm_errors/ms_marco.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
# move to the Approx-Cobweb repo
cd ~/flash/Approx-Cobweb
export PYTHONPATH=$(pwd)

# Default arguments - can be overridden by command line arguments
CONFIG_FILE=${1:-"configs/ms_marco_default.json"}

echo "Starting MS Marco benchmark at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/msmarco.py --config "$CONFIG_FILE"

echo "MS Marco benchmark completed at $(date)"