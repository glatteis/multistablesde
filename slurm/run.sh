#!/bin/bash

#SBATCH --job-name=pygpu
#SBATCH --partition=c18g
#SBATCH --account=thes1544
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
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

# run specified script 
./slurm/$1 ~/artifacts/${SLURM_ARRAY_JOB_ID}_$1/${SLURM_ARRAY_TASK_ID}/
