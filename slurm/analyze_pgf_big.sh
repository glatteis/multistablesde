#!/bin/bash

#SBATCH --job-name=analyze
#SBATCH --account=thes1544
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=08:00:00
#SBATCH --output=/rwthfs/rz/cluster/home/wx133755/output/%x.%A_%4a.out

###
# This script is to submitted via "sbatch" on the cluster.
#
# Set --cpus-per-task above to match the size of your multiprocessing run, if any.
###

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Some initial setup
module purge

cd ~/multistablesde/

# run analyze.py on specified folder
srun python3 -m pipenv run python multistablesde/analyze.py -f $1 --pgf --big
