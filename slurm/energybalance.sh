#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.999 --model=energybalance"

variable_args=(
	"--beta=50"
	"--beta=100"
	"--beta=500"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir ~/artifacts/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/

