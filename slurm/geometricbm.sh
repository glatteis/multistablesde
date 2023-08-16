#!/bin/sh
fixed_args="multistablesde/latent_sde.py --model geometricbm --dt=0.02 --t1=1.0 --decay=0.999 --kl-anneal-iters=50 --latent-size=4"

variable_args=(
	"--beta=0.1"
	"--beta=1"
	"--beta=10"
	"--beta=100"
	"--beta=1000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir ~/artifacts/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
