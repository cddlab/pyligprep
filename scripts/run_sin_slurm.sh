#!/bin/bash
#SBATCH -p q4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --comment=slack:FAIL,END

. /home/apps/Modules/init/profile.sh
module load ambertools26
unset PYTHONPATH

. /path/to/.venv/bin/activate
export TMPDIR="scratch/sin/$SLURM_JOB_ID"
export PYSCF_TMPDIR="$TMPDIR"
export CUPY_CACHE_DIR="$TMPDIR/cupy"
mkdir -p "$TMPDIR"

if [[ "${1:-}" == "--resume" ]]; then
    shift
    exec pyligprep run --resume "sin_run" "$@"
fi
if (( $# > 0 )); then
    echo 'Usage: sbatch $0 [--resume [pyligprep options]]' >&2
    exit 2
fi

pyligprep check -i sin_h.pdb --charge -2 --multiplicity 1
pyligprep run -i sin_h.pdb -o "sin_run" \
    --charge -2 --multiplicity 1 --device gpu --scf-type rks \
    --optimizer berny --method B3LYP-D3BJ --basis '6-31+G(d,p)' \
    --opt-maxcycles 200 --memory-mb 22000 --scf-maxcycle 512 \
    --density 1.0 --qwt1 0.0005 --qwt2 0.001 \
    --max-iterations 100 --tolerance 1e-6 \
    --resp-constraints "sin_resp_constraints.json" \
    --amber-bin /home/apps/ambertools26/bin
