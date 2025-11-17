#!/bin/bash
#SBATCH --job-name=benchmark_qqp
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron
#SBATCH --output=slurm_outputs/qqp.out
#SBATCH --error=slurm_errors/qqp.err
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
CONFIG_FILE=${1:-"configs/qqp_default.json"}

echo "Starting QQP benchmark at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/qqp.py --config "$CONFIG_FILE"

echo "QQP benchmark completed at $(date)"