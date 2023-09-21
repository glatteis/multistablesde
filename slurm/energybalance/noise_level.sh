#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --beta=10 --kl-anneal-iters=1000 --num-iters=10000"

variable_args=(
	"--data-noise-level=0.00"
	"--data-noise-level=0.02"
	"--data-noise-level=0.04"
	"--data-noise-level=0.06"
	"--data-noise-level=0.08"
	"--data-noise-level=0.10"
	"--data-noise-level=0.12"
	"--data-noise-level=0.14"
	"--data-noise-level=0.16"
	"--data-noise-level=0.18"
	"--data-noise-level=0.2"
	"--data-noise-level=0.25"
	"--data-noise-level=0.3"
	"--data-noise-level=0.35"
	"--data-noise-level=0.4"
	"--data-noise-level=0.45"
	"--data-noise-level=0.5"
	"--data-noise-level=0.55"
	"--data-noise-level=0.6"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
