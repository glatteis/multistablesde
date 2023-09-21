#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --beta=10 --kl-anneal-iters=1000 --num-iters=10000"

variable_args=(
	"--data-noise-level=0.20"
	"--data-noise-level=0.22"
	"--data-noise-level=0.24"
	"--data-noise-level=0.26"
	"--data-noise-level=0.28"
	"--data-noise-level=0.30"
	"--data-noise-level=0.32"
	"--data-noise-level=0.34"
	"--data-noise-level=0.36"
	"--data-noise-level=0.38"
	"--data-noise-level=0.4"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
