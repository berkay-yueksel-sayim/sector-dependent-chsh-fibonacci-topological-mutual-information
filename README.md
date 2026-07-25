# Sector-Dependent CHSH Violation in Fibonacci Anyons, and Finite-Size Topological–CHSH Mutual Information in 2D Lattice Models

**Author:** Berkay Yüksel Sayim
**ORCID:** [0009-0004-4993-7352](https://orcid.org/0009-0004-4993-7352)

## Abstract

We quantify the mutual information *I*(*T*;*S*) between topological sector labels
*T* and CHSH values *S* in two-dimensional lattice models. In the ℤ₂ lattice
gauge theory on a 16×16 torus at inverse temperature β = 2.0, we measure
*I* = 0.015 ± 0.011 bits, significant at >2σ in 9 of 10 independent Monte Carlo
runs (corroborated by gauge-invariant plaquette-strip observables). A finite-size
scaling analysis indicates that this sector–CHSH mutual information decays toward
zero with system size. Most significantly, we demonstrate in simulation that
braiding of Fibonacci anyons in the fusion space produces CHSH violation with
|S| = 2.811 (99.4% of the Tsirelson bound), strongly sector-dependent: with
fixed measurement settings the value ranges from near zero to 2.72 depending on
the topological sector phase, while sector-adapted settings achieve CHSH
violation in 99.5% of sectors.

Throughout, "CHSH violation" denotes a CHSH value exceeding 2 on the Fibonacci
fusion space, which does not factorize into spatially separated Alice/Bob
subsystems; the reported values are signatures of topological nonseparability,
not Bell violations in the Einstein–Podolsky–Rosen sense. The ℤ₂ mutual-information
signal is a finite-size effect; the central positive result is a numerical result
within an idealized, zero-temperature, zero-decoherence TQFT model, not a
laboratory measurement.

## Contents

This record is compiled from `main_v2.3.tex` (RevTeX 4-2). The compiled
`main_v2.3.pdf` is included.

## Requirements

The deposited scripts use only NumPy, Matplotlib and Pillow beyond the Python
standard library. The scripts in `data/` import the engines in `sim/` and are
run from their own directory or from the package root; both work.

This release was run and checked with Python 3.12.10, NumPy 2.4.3,
Matplotlib 3.10.8 and Pillow 12.2.0. Other versions are not claimed to be
tested.

## Data and Code Availability

The deterministic analysis scripts and JSON output files deposited with this
record reproduce the lattice results of the paper (the crossover decomposition,
the gauge-invariant confirmation, the finite-size scaling, and the frozen-regime
characterization, with per-cell scaling data in `data/stufe2_analyse.json`).
The Fibonacci values are reproduced from the companion record cited in the
paper; the per-seed Table I entries and the historical model rows of Table VII
have no deposited generating script (see the paper's Data and Code
Availability section for the full scope). All files are archived on Zenodo
under the concept DOI [10.5281/zenodo.19600752](https://doi.org/10.5281/zenodo.19600752)
(which always resolves to the latest version); see License below.

## License
- Paper, figures, and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE`
- Source code (`*.py`): [MIT License](https://opensource.org/licenses/MIT) — see `LICENSE-CODE`
