# Third-party code and scientific provenance

`pyligprep` is distributed under GPL-3.0-only. This is a modified implementation, not an official release of AmberTools, PyRESP, PyPE_RESP, or GPU4PySCF. The code reads neither the installed PyRESP scripts nor their modules at runtime.

## Embedded additive RESP

`src/pyligprep/resp.py` adapts the additive-charge algorithm from the AmberTools 26 distribution of Shiji Zhao's PyRESP (`charge_opt`, `matbld`, `rstran`) and Qiang Zhu's PyRESP_GEN (`getEQU`, `getCH2CH3`, `getSymIdx`, `dumpSymIdx`). The local AmberTools 26 `LICENSE` grants GPL version 3 except for the components listed there; these scripts are not listed as exceptions. Its notice is retained in `licenses/AmberTools-NOTICE.txt`, and the GPL text in `LICENSE`.

PyRESP's upstream repository also supplies the GPL version 3 license:
<https://github.com/ShijiZ/PyRESP/blob/master/LICENSE>.
PyRESP_GEN's upstream is <https://github.com/csu1505110121/pyresp_gen>;
the source used here is the AmberTools 26 distribution, not a license inference
from the absence of a standalone repository LICENSE file.

Original history/credits retained from PyRESP:

- PyRESP 1.0, February 2022: Shiji Zhao.
- RESP 2.4, November 2013, and 2.2, January 2011: q4md-forcefieldtools.org.
- RESP 2.1, October 1994: Jim Caldwell.
- RESP 2.0, September 1992: Christopher Bayly.
- ESPFIT modified version: Ian Gould.
- ESPFIT: U. Chandra Singh and P. A. Kollman.
- PyRESP_GEN 1.0: Qiang Zhu.

Changes made in pyligprep:

- Only the single-molecule, nonpolarizable, additive point-charge model is kept.
- CLI parsing, global state, Fortran namelist parsing, and subprocess execution are replaced by typed Python functions and explicit arrays.
- The matrix reduction for equivalent/frozen atoms is expressed as a linear
  transformation. Total charge is enforced with a Lagrange multiplier.
- PyRESP's harmonic initial solve, hyperbolic radius 0.1, hydrogen exemption,
  and `norm(delta_q) / natoms < 1e-6` convergence criterion are retained.
- Failed or nonfinite fits raise an error, rather than returning a charge file.
- Equivalence uses the prepared chemical graph rather than reconstructing it
  from rounded ESP coordinates. Depth-based equivalence is filtered by element,
  hybridization and immediate neighbor composition, following PyPE_RESP's
  conflict-checking approach. Element identity is never inferred from the first
  character of the atom name. This can deliberately differ from unfiltered
  PyRESP_GEN output for problematic structures. Users can supply explicit
  `stage1`/`stage2` ivary arrays to replace automatic constraints.
- The two-stage reference tests invoke the installed original scripts solely as
  test oracles. They are not production runtime dependencies.

Scientific references:

- Zhao et al., *PyRESP: A Program for Electrostatic Parameterizations of Additive
  and Induced Dipole Polarizable Force Fields*, JCTC 18 (2022), 3654–3670,
  <https://doi.org/10.1021/acs.jctc.2c00230>.
- PyPE_RESP workflow: <https://doi.org/10.1021/acs.jcim.5c00041>.

## ESP surface

`src/pyligprep/esp.py` adapts `unit_surface` and the surface filtering algorithm from GPU4PySCF 1.9.1 `gpu4pyscf/pop/esp.py`, copyright 2021–2024 The PySCF Developers, Apache License 2.0. The license is included in `licenses/Apache-2.0.txt`. Changes: NumPy/SciPy implementation shared by both backends, explicit density units, input validation, batched ESP integrals and limited supported radii. The GAMESS radii and four shell scales are retained. This does not claim to reproduce Gaussian's or QUICK's grid point selection.

## Optimization and restart

The checkpoint validation and restricted/unrestricted density conversion were adapted from the user's existing `ligcl3_opt.py`. Optimization defaults, tight thresholds, density fitting, Newton SCF, and gradient grid response were carried over into the package. The existing source file is unchanged.

Exact source file SHA-256 values and the inspection date are recorded in `licenses/source-provenance.json`. No Amber force-field databases or executables are redistributed; an installed AmberTools 26 is required for export.
