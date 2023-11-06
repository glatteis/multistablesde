# The multistablesde package

This package contains all experiments performed in my Master's thesis "Neural Stochastic Differential Equations for Multistable Dynamical Systems". Install `pipenv` and run `pipenv install` to install the dependencies and the package.

The experiments are included as `slurm` scripts, you can run them as follows:

    cd slurm
    # Edit "account" and "output" folders (sorry, they are just hardcoded as mine) and other slurm variables you need to tweak
    vim run.sh
    sbatch --array=0-<number of jobs> run.sh <file>

Example:

    cd slurm
    sbatch --array=0-7 run.sh energybalance/betas.sh

You can analyse the results by identifying the correct folder in `~/artifacts` and running `analyze.sh`:

    cd slurm
    # for matplotlib pdfs
    sbatch analyze.sh ~/artifacts/<output folder>/
    # for pgfs (latex)
    sbatch analyze_pgf.sh ~/artifacts/<output folder>/
