# pyligprep

English | [日本語](README_ja.md)

A Python package that adds hydrogens to a ligand PDB, optimizes its geometry with PySCF/GPU4PySCF, computes the HF ESP, fits charges with a built-in two-stage RESP, and generates GAFF2 parameters with AmberTools 26. The outputs are `ligand.mol2`, `ligand.prep` (PREPI format), and `ligand.frcmod`, all preserving the input atom names. With `--resname SIN` (`-r SIN`) on `run` / `export`, the files are named `SIN.*` and the residue name in the MOL2 and PREP files becomes `SIN`. Upper and lower case are preserved as given.

The first release targets organic ligands that are a single molecule, single residue, and single conformer
(H, C, N, O, F, P, S, Cl). Metals, salt mixtures, multiple models, and unresolved altlocs are rejected.
It does not rebuild missing heavy atoms, predict pKa, search conformers, or confirm minima by frequency analysis.

## Installation

Requires Python 3.12 or later. The primary target is Linux x86_64 with GPUs and Slurm (developed and validated on Ubuntu 24.04).
On macOS (Apple Silicon), geometry optimization and parameter generation for small molecules work on the CPU only (see below). The GPU extra is for Linux x86_64 / CUDA 12.

```bash
cd pyligprep
uv sync --extra cuda12
module load ambertools26 # puts the AmberTools bin directory on PATH
unset PYTHONPATH
source .venv/bin/activate
pyligprep doctor
```

For CPU only, use `uv sync`. To install from the local checkout into an existing Python environment, use `uv pip install '.[cuda12]'` (`uv pip install .` for CPU only).
The Open Babel wheel (`openbabel-wheel`), which `hadd` uses to perceive bond orders from a PDB alone, is installed by default and run as the separate `obabel` program. Hydrogen addition with a SMILES string or template does not use Open Babel.

The extras pin GPU4PySCF 1.9.1, CuPy CUDA12 13.4.1, cuTENSOR 2.2.0, and GPU libxc 0.9.0. The resolved development environment is recorded in `uv.lock`. Nothing depends on existing hand-built environments. The AmberTools 26 binaries, an NVIDIA driver, and a compatible CUDA runtime must be provided separately. The AmberTools module on the development machine loads CUDA 12.8, so do not combine it with CUDA 12.9. For other environments, change the module settings in the bundled Slurm scripts.

### macOS (Apple Silicon, CPU only)

Install AmberTools 26 from the Homebrew `brewsci/bio` tap. Also install Homebrew LLVM (`llvm`) so that PySCF is built
with OpenMP (**recommended and the default**).

```bash
brew install brewsci/bio/ambertools llvm
export AMBERHOME=/opt/homebrew/opt/ambertools
cd pyligprep
uv sync
source .venv/bin/activate
pyligprep doctor
```

The setup is complete when `doctor` reports `gpu.available` as `false`, `dispersion.d3_backend` as `simple-dftd3`,
and three commands under `amber`. `pyligprep` uses `antechamber` and the other tools found on PATH
(Homebrew's `/opt/homebrew/bin`). `AMBERHOME` is read by AmberTools itself and by the tests. Do not request the `cuda12` extra, which is Linux only.

#### PySCF with OpenMP (recommended)

PyPI's macOS PySCF wheels are built without OpenMP, so `--threads` has no effect (a `OpenMP is not available` warning is printed and PySCF runs on one thread). Therefore `[tool.uv.sources]` and `[tool.uv.extra-build-variables]` in `pyproject.toml` make `uv sync` on macOS build PySCF from source with the Homebrew LLVM clang and OpenMP enabled (Linux keeps using the PyPI wheel).

- The first `uv sync` spends several minutes building PySCF. Later runs use the uv cache.
- Apple's clang does not support OpenMP and silently builds without it, so the LLVM clang is specified.
  The path assumes Homebrew on Apple Silicon (`/opt/homebrew/opt/llvm`). The build fails if `llvm` is not installed.
- The PyPI sdist of PySCF 2.15.0 lacks `libxcfun.patch`, which the build needs, so it fails. The source is therefore fetched from
  the GitHub release archive (v2.15.0). This URL pins PySCF on macOS to 2.15.0; update the URL when upgrading.
- These settings are uv only (effective for `uv sync` and `uv pip install`). A plain `pip install` without uv installs the wheel without OpenMP.
- Check whether OpenMP is enabled as follows (it is enabled if `8` is printed with no warning).

  ```bash
  python -c "from pyscf import lib; lib.num_threads(8); print(lib.num_threads())"
  ```

Optimizing SIN (12 atoms) took 98 s with the wheel and 44 s with `--threads 8` (identical output files). To go without OpenMP, delete the two sections above from `pyproject.toml` and run `uv lock` again.

- D3(BJ) dispersion: Linux uses `pyscf-dispersion`, but that package has no macOS wheel.
  On macOS, `dftd3`, the Python binding of the same simple-dftd3 library, is used automatically (selected per
  platform in `pyproject.toml`). DFT-D4 is not available on macOS.
- `--device auto` falls back to the CPU (with a warning). You may also pass `--device cpu` explicitly.
  `--device gpu` stops with an error.
- The default `--memory-mb 22000` may exceed the physical memory. Set `--memory-mb` to match the installed memory.
- As a guide, set `--threads` to about the number of physical cores (on Apple Silicon, the number of performance cores). Compute large molecules on Linux/GPU.
- Do not use the Slurm scripts; run the commands directly.
- Resuming on macOS (`--resume`) a stage created on Linux, even for the same molecule, does not reuse completed results
  because the dependency versions differ; they are recomputed.

Example of the full workflow with the bundled succinate dianion (`examples/sin_h.pdb`, nearly optimized coordinates):

```bash
pyligprep run -i examples/sin_h.pdb -o sin_mac \
  --charge -2 --multiplicity 1 --smiles 'O=C([O-])CCC(=O)[O-]' \
  --device cpu --threads 8 --memory-mb 8000 --resname SIN
```

This writes `sin_mac/SIN.{mol2,prep,frcmod}` and `sin_mac/versions.json`; the working directory `sin_mac/tmp/` is removed after success.
`examples/` does not ship sidecar JSON files, so pass the same `--smiles` as for `hadd`.

## Commands, inputs, and outputs

The command examples below assume the virtual environment is activated.
Otherwise, replace `pyligprep` with `.venv/bin/pyligprep`. Submit the bundled Slurm scripts from `pyligprep/`, the directory containing `pyproject.toml`. The charge −1 and singlet in the examples should be changed to match your ligand.

| Subcommand | Purpose | Main input → output |
| --- | --- | --- |
| `doctor` | Detect dependencies, GPU, and AmberTools | Runtime environment → JSON on standard output |
| `hadd` | Add hydrogens and write a PDB for review | PDB → hydrogenated PDB + sidecar JSON |
| `check` | Check consistency of atom names, bonds, charge, and multiplicity | Hydrogenated PDB → validation result |
| `opt` | Geometry optimization (step A) | Hydrogenated PDB → optimization stage |
| `resp` | HF/ESP calculation and two-stage RESP (step B) | PDB or completed opt stage → RESP stage |
| `export` | GAFF2 atom typing and charge rounding with total-charge correction (step C) | Completed RESP stage → MOL2, PREP, FRCMOD |
| `run` | Run A → B → C in sequence | Reviewed hydrogenated PDB → results of all three steps |

Check the environment and the arguments of each command as follows.

```bash
pyligprep --version   # same as -V; e.g. pyligprep 0.0.1
pyligprep --help
pyligprep opt --help
pyligprep resp --help
pyligprep export --help
pyligprep doctor --amber-bin /home/apps/ambertools26/bin
```

Every subcommand accepts `-V` / `--version`, and the first line of `--help` also shows the version, such as `pyligprep 0.0.1`.
The number is `project.version` in `pyproject.toml`, read through the installed package metadata.
After changing the version, reinstall with `uv sync` for the display to follow.

`doctor` inspects the current node. On a login node without an allocated GPU,
it may report the GPU as unavailable. `doctor` and `check` never start a geometry optimization.

| Argument | Meaning and when it applies |
| --- | --- |
| `-i FILE` / `--input FILE` | Input PDB. Used by `hadd`, `check`, `opt`, `resp`, and `run` |
| `-o PATH` / `--output PATH` | Output PDB name for `hadd`. A **new output directory** for `opt`, `resp`, `export`, and `run` |
| `--charge INTEGER` | Integer total charge of the molecule. Required when starting `check`, `opt`, `resp`, or `run` from a PDB |
| `--multiplicity INTEGER` | Spin multiplicity 2S+1: 1 for a singlet, 2 for a doublet. Required under the same conditions as above |
| `--from DIRECTORY` | Stage directory of a completed opt for `resp`, or of a completed RESP for `export` |
| `--resume DIRECTORY` | Resume the output directory of the same subcommand from the working files in its `tmp/` |
| `--keep-tmp` | `run` only. Keep `tmp/` after success (by default it is removed). It is always kept when `run` stops midway |
| `--smiles STRING` / `--template FILE` | Assign bond orders and formal charges to the PDB. Specify only one of them |
| `--amber-bin DIRECTORY` | AmberTools bin directory. Used by `doctor`, `export`, and `run`. Searched on PATH when omitted |
| `-r NAME` / `--resname NAME` | Output file prefix and residue name for `run` and `export`. Case-sensitive. 1–3 ASCII alphanumeric characters starting with a letter |

Do not combine `-i`, `--from`, and `--resume`. A new calculation needs `-o`; when resuming, `--resume` also identifies the output, so `-o` is not needed.
Pass `--from` / `--resume` the output directory (`-o`) of an earlier stage, not the path of an XYZ file or a checkpoint. Directories written by older versions (with `manifest.json` / `attempts/`) cannot be resumed or used with `--from`; rerun into a new output directory. The saved total charge and multiplicity cannot be changed afterwards. To change the electronic state, start again from the PDB into a new output directory.

## 1. Add hydrogens and review

```bash
pyligprep hadd -i ligand.pdb -o ligand_h.pdb
```

The outputs are `ligand_h.pdb` and `ligand_h.pyligprep.json`. The names, order, and coordinates of the original atoms are preserved, and new hydrogens are appended at the end with unique names. Existing hydrogens are not removed. The JSON stores bond orders, formal charges, the SMILES, and the parent atom of each added hydrogen.

**Review the structure and protonation state yourself before running opt or run.**

Bond orders and protonation states derived from a PDB alone are guesses. To specify the intended state,
use a SMILES string or a single-molecule SDF/MOL/MOL2 template, for example:

```bash
# This SMILES is an acetate ion example. Replace it with the SMILES of your ligand.
pyligprep hadd -i acetate.pdb -o acetate_h.pdb \
  --smiles 'CC(=O)[O-]' --charge -1

pyligprep hadd -i ligand.pdb -o ligand_h.pdb \
  --template ligand.sdf --charge -1
```

`hadd --charge` is a check, not a feature that chooses protonation sites to reach the given total charge. It stops if the predicted charge differs. `run` never adds hydrogens either. Atom names in the input PDB must be unique within the residue. Atom names must not start with a digit (e.g. `1H1`, `2H10` written by GaussView and similar tools), because antechamber cannot infer their elements when writing the MOL2; `hadd` / `check` / `opt` / `run` reject such input, so rename them (e.g. `H1A`) first.

After editing a PDB, its hash no longer matches the sidecar. If you fix the structure, save the fixed PDB under a different name and run `hadd` again with a suitable template. Protonation changes that require removing existing hydrogens must be made in the input first.

For a real ligand handled during development (ligcl3), the state inferred from the PDB alone was C30H48O3 with charge 0, which differed from the intended state (C30H45O3, charge −1). If a specific protonation state is intended, always supply the corresponding chemical template. See the [validation record](docs/validation.md) (in Japanese) for details.

## 2. Check before computing

```bash
pyligprep check -i ligand_h.pdb --charge -1 --multiplicity 1
```

The total charge and multiplicity are required when starting `check`, `opt`, `resp`, or `run` from an input PDB. With `--from` and `--resume`, they are inherited from the validated saved data.
A PDB that already has all hydrogens can skip `hadd`. If bond perception fails, specify `--smiles` or `--template`.

## 3. Individual steps

### A. Geometry optimization: `opt`

```bash
pyligprep opt -i ligand_h.pdb -o work/opt \
  --charge -1 --multiplicity 1 --device gpu
```

The default is Berny with B3LYP-D3BJ/6-31+G(d,p). Example of changing the algorithm and settings:

```bash
pyligprep opt -i ligand_h.pdb -o work_geometric/opt \
  --charge -1 --multiplicity 1 --device gpu \
  --optimizer geometric --method B3LYP-D3BJ --basis '6-31+G(d,p)' \
  --max-steps 400 --threads 8 --memory-mb 22000
```

`--optimizer berny` selects Berny explicitly. Quote basis set names containing parentheses, as above. `--method` and `--basis` apply to the optimization only and do not affect the HF/6-31G* of the later ESP calculation. The final structure is written as `work/opt/optimized.xyz` and `work/opt/optimized.pdb`. Use `--from work/opt` for the next step.

### B. ESP and two-stage RESP: `resp`

```bash
pyligprep resp --from work/opt -o work/resp --device gpu
```

This inherits the atom names, bonds, full-precision coordinates, total charge, and multiplicity from the optimization stage.
The device and thread count can also be specified for this command.
To set the RESP iteration limit or reviewed equivalence constraints:

```bash
pyligprep resp --from work/opt -o resp_custom --device gpu \
  --max-iterations 100 --resp-constraints reviewed_constraints.json
```

`reviewed_constraints.json` must be an existing file with `stage1` / `stage2` arrays in atom order.
See "Resuming and saved data" below for the format.
ESP and RESP run in one subcommand, and each intermediate result is saved separately in `tmp/`.
The output is `work/resp/charges.json`: the atom names, the final RESP charges, and the iterations and RMS ESP error of each stage.

### C. Writing MOL2, PREP, and FRCMOD: `export`

```bash
pyligprep export --from work/resp -o parameters

# When the AmberTools commands are not on PATH
pyligprep export --from work/resp -o parameters_explicit \
  --amber-bin /home/apps/ambertools26/bin
```

`export` uses `antechamber`, `parmchk2 -s gaff2`, and `tleap`.
If they are not on PATH, pass `--amber-bin /path/to/ambertools/bin`.
The outputs are `parameters/ligand.mol2`, `parameters/ligand.prep`, and `parameters/ligand.frcmod`.
It verifies the atom names, order, charges, and bonds of the MOL2, and the names, charges, and bonds of the real atoms in the PREPI.
The PREPI contains dummy atoms and a formal reordering. LEaP builds a topology from each format,
and the step does not succeed if there are unresolved parameters or errors.

To set the file name and residue name, add `--resname` or `-r`:

```bash
pyligprep export --from work/resp -o sin_parameters --resname SIN
```

This example creates `sin_parameters/SIN.mol2`, `SIN.prep`, and `SIN.frcmod`.
The MOL2 MOLECULE name, the residue name in the ATOM lines, and the SUBSTRUCTURE name all match the residue name in
the PREP's `SIN INT 0` line. FRCMOD has no residue name field, so the name is reflected in the file name only.
With `--resname SiN`, both the file names and the residue name become `SiN`.
When omitted, the files are named `ligand.*` and the residue name comes from the input PDB.
Atom names, coordinates, and RESP results are not changed.

Output charges are rounded to five decimal places, with the sixth digit written as 0 (e.g. `0.047200`).
Exact halves are rounded away from zero. The resulting drift in the total is corrected
by integer arithmetic in units of `0.000010 e`, so the sum of charges in the file equals the specified integer exactly.
Groups with charges near 0, −1, or +1 are preferred, and both the stage 2 RESP equivalences and the stage 1 equivalences
among atoms fixed in stage 2 are preserved. Unrelated atoms are never treated as equivalent merely
because their values happened to be equal.

It first searches for combinations that change each group by a single unit, skipping groups that cannot
absorb the residual. If needed, it moves several groups in opposite directions or by several units, keeping the change small.
As an exception, for molecules such as a +1 molecule whose atoms all belong to equivalence groups of three,
an integer total that keeps both five-digit precision and the equivalences is mathematically impossible. In that case
it stops with an explicit error rather than breaking the equivalences.

The RESP charges before correction, the equivalence groups, the rounded values, and the number of correction units are recorded in `tmp/charge-rounding.json`.
The original RESP results are not modified. The internal format of `.prep` and the antechamber
`-fi` / `-fo` option remain `prepi`. The real-atom charges are identical in the MOL2 and the PREP.

Five-digit rounding and total-charge correction are applied automatically by `export` / `run`; no extra flag is needed.
`-o` specifies an output directory; it is not for a file name such as
`-o ligand.mol2`. Every command writes only its deliverables and `versions.json` directly under it,
and keeps all other files in `tmp/` inside it:

```
parameters/                 # -o of export (or run)
├── ligand.mol2             # deliverables
├── ligand.prep
├── ligand.frcmod
├── versions.json           # pyligprep, Python, Python packages, and the AmberTools binaries used
└── tmp/                    # working files, logs, and the state used by --resume / --from
```

| Command | Deliverables directly under `-o` |
| --- | --- |
| `opt` | `optimized.pdb`, `optimized.xyz` |
| `resp` | `charges.json` |
| `export` / `run` | `ligand.mol2`, `ligand.prep`, `ligand.frcmod` (the `--resname` value replaces `ligand`) |

| File | Contents |
| --- | --- |
| `ligand.mol2` | Atom names and order from the input PDB, optimized coordinates, GAFF2 types, corrected RESP charges |
| `ligand.prep` | PREPI format converted from the MOL2. Real-atom names and charges match the MOL2 |
| `ligand.frcmod` | Supplementary parameters written by `parmchk2 -s gaff2` |
| `versions.json` | Names of the current deliverables (files under an earlier `--resname` may remain beside them), and versions of pyligprep, Python, and the Python packages; for `export` / `run`, also the path, SHA-256, and AmberTools release of `antechamber` / `parmchk2` / `tleap` |
| `tmp/charge-rounding.json` | Original RESP charges, equivalence groups, and the rounding and total-charge correction record |
| `tmp/atom-map.json` | Atom number, name, type, and charge of each atom in the output MOL2 |
| `tmp/state.json` | Input, settings, dependency versions, status, and results of the stage, used by `--resume` / `--from` |

### Starting from an already optimized structure

If the structure was already optimized with another program, go straight from the PDB to step B.

```bash
pyligprep resp -i optimized.pdb -o work/resp \
  --charge -1 --multiplicity 1 --device gpu
```

This entry point is a single-point calculation on the given structure and does not check whether it is optimized.
`resp --from work/opt` verifies that the optimization converged.
Between steps, the unrounded coordinates are used rather than the rounded PDB coordinates, so use `--from` in principle.
`-i` takes a PDB. There is currently no CLI to read an arbitrary XYZ file directly, or to merge
the atom names of a PDB with XYZ coordinates from a separate file.

Rewriting only the outputs from a saved RESP does not recompute HF/ESP/RESP either.

```bash
# Resume the existing export, rewriting only when needed
pyligprep export --resume parameters

# Write to a different, new directory
pyligprep export --from work/resp -o parameters_new
```

The CLI writes only `ligand.mol2`, `ligand.prep`, `ligand.frcmod` (or the `--resname` names), and `versions.json`
directly under the given directory. Copies placed elsewhere are not updated.

## 4. Batch runs and Slurm

```bash
pyligprep run -i ligand_h.pdb -o work \
  --charge -1 --multiplicity 1 --device gpu
```

This runs the optimization, RESP, and export in order in `work/tmp/opt`, `work/tmp/resp`, and `work/tmp/export`.
After success, only `work/ligand.{mol2,prep,frcmod}` and `work/versions.json` remain, and `work/tmp/` is removed.
Add `--keep-tmp` to keep `work/tmp/`, for example to inspect the logs or to rerun later with different settings using `run --resume`.
If `run` stops midway, `work/tmp/` is always kept, and `run --resume work` continues from the step that stopped.
`run` also accepts `--resname` / `-r`. For example, the following command
writes `sin_named_run/SIN.{mol2,prep,frcmod}`.

```bash
pyligprep run -i sin_h.pdb -o sin_named_run \
  --charge -2 --multiplicity 1 --device gpu --resname SIN
```

It uses the same Python functions as the individual subcommands and does not launch PyRESP's external Python scripts.
`hadd` is not part of the batch run. Pass `run` a hydrogenated PDB you have reviewed.
Optimization options such as `--optimizer` / `--method` / `--basis` and RESP options such as
`--max-iterations` / `--resp-constraints` can be given together.

The bundled Slurm scripts are examples for the machine used in development (partition `q4`, modules under `/home/apps`).
On other systems, change the partition, module names, initialization path, and virtual environment path.

| Script | Purpose | Requested resources |
| --- | --- | --- |
| `scripts/run_sin_slurm.sh` | Example running `check` and `run` for the succinate dianion (SIN) | 8 tasks × 1 CPU, 7 GB per CPU, no GPU requested, no time limit set |
| `scripts/test_gpu_slurm.sh` | Tests with the `gpu` marker (AmberTools and reference tests enabled too) | 1 GPU, 4 CPUs, 24 GB memory, 1 hour limit |

`run_sin_slurm.sh` reads `sin_h.pdb` and `sin_resp_constraints.json` from the directory given by the environment variable `project`,
and writes to `$project/sin_run`. The virtual environment path (`/home/moriwaki/...`) and `--amber-bin` are hard-coded in the script.
It passes `--device gpu` to `run`, but the script itself does not request a GPU. It stops if placed on a node without a GPU,
so submit it with an `sbatch` option that requests one.

```bash
project=$PWD/sin_work sbatch --gpus=1 scripts/run_sin_slurm.sh

# Resume a calculation that stopped midway (arguments after --resume are passed to pyligprep run)
project=$PWD/sin_work sbatch --gpus=1 scripts/run_sin_slurm.sh --resume --max-steps 400
```

Slurm's standard output and standard error go to `run_sin_slurm.sh.JOBID.out` / `.err`.
Submit `test_gpu_slurm.sh` from the package directory with `sbatch scripts/test_gpu_slurm.sh`; it writes temporary files and
coverage data to `validation/`. It uses `--cov`, so add `pytest-cov` to the virtual environment (it is not in the dev dependencies).

Do not submit several jobs for the same output directory at the same time.
`--threads` and `--memory-mb` are application settings and do not change the Slurm
allocation. To change the allocation too, put `sbatch` options before the script name.

## Calculation settings

| Item | Default / how to set |
| --- | --- |
| Optimization | Berny, B3LYP-D3BJ/6-31+G(d,p) |
| Algorithm | `--optimizer berny` or `--optimizer geometric` |
| DFT | `--method`, `--basis` |
| Spin | RKS for closed-shell singlets, UKS otherwise. Can be set with `--scf-type rks` / `uks` (`opt` / `run` only) |
| SCF | DFT uses density fitting and Newton SCF (at most 100 cycles), integration grid level 3 |
| Convergence | SCF energy 1e-10, gradient 1e-6. Berny: gradientmax 1.5e-5, gradientrms 1e-5, stepmax 6e-5, steprms 4e-5 (atomic units) |
| Steps | `--max-steps 200` |
| Initial trust radius | `--opt-trust N`: positive integer on the Gaussian `Opt=MaxStep=N` scale; omitted by default |
| geomeTRIC | GAU_TIGHT, analytical initial Hessian, trust=0.05, tmax=0.10 |
| Geometry constraints | geomeTRIC only: `--constraints constraints.txt`, atom numbers start at 1 |
| Device | `--device auto` / `gpu` / `cpu` |
| Resources | `--threads 8 --memory-mb 22000 --max-scf-cycles 512` |
| ESP | Gas-phase HF/6-31G*, Cartesian d functions, RHF for closed shell / UHF for open shell, no density fitting |
| ESP grid | GAMESS radii from GPU4PySCF 1.9.1, scale factors 1.4/1.6/1.8/2.0, `--density 1.0` points/Å² |
| RESP | Restraint weights `--qwt1 0.0005 --qwt2 0.001`, hyperbola parameter 0.1, hydrogens unrestrained |
| RESP convergence | `--tolerance 1e-6 --max-iterations 42`, same criterion as PyRESP |
| RESP equivalence | `--equivalence-depth 3` (1–5); explicit constraints with `--resp-constraints FILE` |

Give optimization options to `opt` / `run`, and ESP/RESP options to `resp` / `run`.
`--device` / `--threads` / `--memory-mb` / `--max-scf-cycles` are
common to `opt` / `resp` / `run`. `export` generates files from saved charges, so it
needs no GPU and does not accept these quantum-chemistry options.
`--max-scf-cycles` applies as given to the HF for the ESP, but the DFT Newton SCF is capped at 100.
The HF for the ESP is RHF for a closed-shell singlet and UHF otherwise, regardless of `--scf-type`.

Berny's internal-coordinate thresholds and geomeTRIC's Cartesian thresholds are not the same thing.

`--opt-trust N` controls the **initial geometry trust radius**, using the input scale of Gaussian 16's
`Opt=MaxStep=N`: `0.01 * N` in Bohr/radian for Berny's internal coordinates. For example, `--opt-trust 10`
sets Berny's initial radius to `0.1`, and `--opt-trust 30` to `0.3`. Only positive integers are accepted.
For geomeTRIC, the length is converted to Å: `--opt-trust 10` gives approximately `0.052918 Å`.
geomeTRIC uses a Cartesian displacement metric, so this conversion does not make the two optimizers or
their steps equivalent to Gaussian. Its `tmax` remains `0.10 Å` unless the requested initial radius is
larger; in that case `tmax` is raised to the initial radius to prevent silent clipping.

When omitted for a new calculation, existing defaults are kept: Berny `0.3` and geomeTRIC `0.05 Å`
with `tmax=0.10 Å`. On resume, an omitted `--opt-trust` keeps the saved value; an explicit value overrides
it, and older saved states without this option use the existing defaults. The optimizer may adjust the
radius during optimization. This option does not set a fixed maximum step, the geometry convergence
thresholds (`stepmax` / `steprms`), the iteration count (`--max-steps`), or the SCF orbital rotation step.

`auto` falls back to the CPU when GPU initialization is unavailable.
SCF non-convergence, out-of-memory, and other failures after a GPU start are never hidden by rerunning on the CPU.
With `gpu`, it stops if no GPU is available.

For open-shell HF, the ESP is built from the α+β density. For electronic states that differ from the usual closed-shell AMBER/RESP procedure,
the applicability of the charges and force field must be checked separately.

## Resuming and saved data

Resuming reloads the original input and settings and changes only the settings given in addition.
Repeating the same `-i ... -o ...` as a new run against an existing directory stops without overwriting.

```bash
# Continue an optimization that failed for lack of steps, from the saved structure
pyligprep opt --resume work/opt --max-steps 400

# Switch the optimization algorithm, starting from the same structure
pyligprep opt --resume work/opt --optimizer geometric

# Restart from the saved geometry with a specified initial trust radius (Gaussian MaxStep=10 scale)
pyligprep opt --resume work/opt --opt-trust 10

# If the RESP fit ran out of iterations, continue while reusing the completed ESP
pyligprep resp --resume work/resp --max-iterations 100

# Resume or refresh only the outputs
pyligprep export --resume parameters

# Write the outputs from the saved RESP under a different file and residue name
pyligprep export --resume parameters -r SIN

# For a batch run, give its output directory (tmp/ must remain: after a failure, or with --keep-tmp)
pyligprep run --resume work
```

Give the stage of the same subcommand. For example, use `opt --resume work/opt` and `resp --resume work/resp`, not `opt --resume work` or `resp --resume work/opt`. To resume on a GPU node, submit a job script containing the command above with `sbatch` (the bundled `run_sin_slurm.sh` is an example specific to SIN).

If RESP is redone and the charges change, an existing `export --resume parameters` still uses the earlier RESP saved in that export. To use the new RESP, give a new output directory, as in `export --from work/resp -o parameters_updated`. `run --resume work` checks each child stage in `work/tmp/` in turn and passes the latest RESP result to export.

If `--resname` is omitted with `--resume`, the name saved in the stage is kept. Giving it explicitly switches to the new name. Even `run --resume work --resname SIN` (with `tmp/` kept by `--keep-tmp`) reuses the completed optimization, ESP, and RESP when the settings are unchanged, and updates only the export step. Files under the previous name are kept. Hand-edited outputs, including the copies directly under the `run` output directory, are never overwritten and cause a stop, so to regenerate under the same name use `export --from ... -o new-directory --resname SIN`.

- `tmp/state.json` stores the input, settings, dependency versions, status, and the record of each completed substep.
- There is no attempt history. A resume continues in the same `tmp/`, and logs are overwritten.
- The optimization saves `last.xyz`, `last.json`, the SCF checkpoint, and the trajectory (appended across resumes).
- The RKS/UKS checkpoint format is detected, the density is converted, and the SCF is reconverged.
  UKS→RKS discards the spin polarization. The optimizer's internal history is not restored.
- When the method or basis changes, a fresh SCF is run from the latest coordinates.
- HF, ESP, stage 1 RESP, and stage 2 RESP are saved separately, and a resume continues after the last completed one.
  A change to the fit alone reuses the ESP.
- SCF checkpoints (`*.chk`) are deleted when a stage completes, to keep the output small. A change to the ESP grid
  after the RESP stage has completed therefore reruns the HF.
- When the structure changes, `resp` / `export` clear their `tmp/` and recompute for that structure.
- Completed results are reused after their hashes are verified. Intermediate structures are never treated as converged.
- Concurrent writes to the same stage are rejected by a lock, and user-edited final outputs are never overwritten.

`resp` stores `constraints.json`, `esp.npz` (Bohr / atomic units), `esp.dat`, `stage1.in`, `stage2.in`, and the charge JSON of each stage in `tmp/`. The `.in` files are for review and comparison and are not executed. Automatic equivalence detection follows PyRESP_GEN and checks consistency of elements, hybridization, and bonded neighbors. Review the constraints for large molecules or unusual functional groups.
`--resp-constraints constraints.json` gives the `stage1`/`stage2` arrays.
Each element is `0 = free`, `-1 = fixed`, or `a positive integer = the 1-based number of an earlier atom that must have the same charge`.
The arrays follow the atom order of the input PDB. The total-charge constraint is always applied.

## Development, validation, and distribution

```bash
uv sync --extra cuda12
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync ty check src tests
uv run --no-sync pytest -q
uv build
```

On macOS, use `uv sync` (`cuda12` is Linux only; PySCF is built from source with OpenMP).
The default test run skips tests that need a GPU, AmberTools, or the original PyRESP. The CPU HF/ESP and the D3(BJ) dispersion values and gradients are tested, but no geometry optimization is run on the CPU.
On macOS, the `PYLIGPREP_TEST_AMBER=1` / `PYLIGPREP_TEST_REFERENCE=1` tests can also run with Homebrew's AmberTools.

The tests look for AmberTools in `PYLIGPREP_AMBER_BIN=/path/to/ambertools/bin`, then `$AMBERHOME/bin`, then PATH.
These environment variables are for the tests only; the CLI uses only `--amber-bin` or the commands on PATH.

See [docs/validation.md](docs/validation.md) (in Japanese) for the validation results. Wheels and sdists are built into `dist/`.

Copyright (c) 2026, Yoshitaka Moriwaki, CDD Laboratory.
pyligprep is licensed under the GNU General Public License, version 3 only (GPL-3.0-only; [LICENSE](LICENSE)).
The RESP implementation is adapted from AmberTools 26 PyRESP / PyRESP_GEN (GPL-3.0-only),
and the ESP grid and GAMESS radii from GPU4PySCF 1.9.1 (Apache-2.0); the sources are credited at the top of
`src/pyligprep/resp.py` and `src/pyligprep/esp.py`. The AmberTools binaries and databases are not bundled.
Open Babel (`openbabel-wheel`, GPL-2.0) is installed as a dependency but is neither imported nor linked:
`hadd` runs its bundled `obabel` program as a separate process.
