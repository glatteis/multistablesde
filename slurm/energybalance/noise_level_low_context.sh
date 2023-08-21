#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --beta=10 --context-size=4"

variable_args=(
	"--data-noise-level=0.06"
	"--data-noise-level=0.08"
	"--data-noise-level=0.10"
	"--data-noise-level=0.12"
	"--data-noise-level=0.14"
	"--data-noise-level=0.16"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
