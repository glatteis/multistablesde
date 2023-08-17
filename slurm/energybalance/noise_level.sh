#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.999 --model=energy --beta=1000"

variable_args=(
	"--data-noise-level=0.01"
	"--data-noise-level=0.75"
	"--data-noise-level=0.10"
	"--data-noise-level=0.125"
	"--data-noise-level=0.15"
	"--data-noise-level=0.175"
	"--data-noise-level=0.2"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
