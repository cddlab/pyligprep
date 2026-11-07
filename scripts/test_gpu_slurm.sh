#!/bin/bash
#SBATCH -p q4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --time=01:00:00

set -euo pipefail
. /home/apps/Modules/init/profile.sh
module load ambertools26
unset PYTHONPATH
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYLIGPREP_TEST_GPU=1
export PYLIGPREP_TEST_AMBER=1
export PYLIGPREP_TEST_REFERENCE=1
export TMPDIR="$PWD/validation/scratch-$SLURM_JOB_ID"
export PYSCF_TMPDIR="$TMPDIR"
export CUPY_CACHE_DIR="$TMPDIR/cupy"
export COVERAGE_FILE="$PWD/validation/.coverage.gpu-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
.venv/bin/python -m pytest -v -m 'gpu' --cov=pyligprep --cov-report=term --basetemp "$PWD/validation/gpu-$SLURM_JOB_ID"
