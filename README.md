# CHSH-Form Values Above 2 for Standard Fibonacci Anyons, and the Non-Topological Origin of Finite-Size Sector-CHSH Mutual Information in Z2 Lattice Gauge Theory

**Author:** Berkay Yüksel Sayim
**ORCID:** [0009-0004-4993-7352](https://orcid.org/0009-0004-4993-7352)

## Abstract

We quantify the mutual information *I*(*T*;*S*) between topological sector labels
*T* and CHSH values *S* in two-dimensional lattice models. In the ℤ₂ lattice
gauge theory on a 16×16 torus at inverse temperature β = 2.0, we measure
*I* = 0.0173 bits after subtracting the null level, 95% interval [0.0159, 0.0187],
significant above 3σ in all 21 seeds against a circular-shift null whose
false-positive level we measure at 0.05 of 21. A finite-size
scaling analysis indicates that this sector–CHSH mutual information decays toward
zero with system size. Most significantly, in simulation the two-generator d1b
circuit model of Fibonacci anyons produces CHSH-form values above 2: |S| =
2.7334 (96.64% of the Tsirelson bound) at δ = 0, the point that realizes the
standard Fibonacci modular tensor category, and |S| = 2.8114 (99.40%) at δ ≈
1.44π, the maximum of a 50-point grid over a phase deformation δ of the braid
representation; for δ ≠ 0 the representation does not realize the standard
category. The values depend strongly on δ: with fixed measurement settings
they range from near zero to 2.72, while settings adapted to each δ give |S| >
2 at 99.5% of the δ values sampled.

Throughout, "CHSH violation" denotes a CHSH value exceeding 2 on the Fibonacci
fusion space, which does not factorize into spatially separated Alice/Bob
subsystems; the reported values are signatures of nonseparability, not Bell
violations in the Einstein–Podolsky–Rosen sense. The ℤ₂ mutual-information
signal is a finite-size effect; the central positive result is a numerical result
within an idealized, zero-temperature, zero-decoherence TQFT model, not a
laboratory measurement.

## Contents

This record is compiled from `main_v2.4.tex` (RevTeX 4-2). The compiled
`main_v2.4.pdf` is included.

## Requirements

The deposited scripts use only NumPy, Matplotlib and Pillow beyond the Python
standard library. Scripts in `data/` that need the superseded engines import them from `sim/`; all scripts are
run from their own directory or from the package root; both work. The one
exception is `data/fibonacci_enumeration.py`, which imports its gate
definitions from the companion record instead; run it with `--engine
<path>` if the two records are not unpacked side by side.

`data/stufe2_analyse.py` additionally requires SciPy
(`scipy.optimize.curve_fit`) for the three pre-registered model fits.

Naming note: the paper calls the null hypothesis of Sec. II E the *permutation
null* and writes I_shuffled for the surrogate; `stufe2_analyse.py` and the
deposited `stufe2_analyse.json` call the same quantity *scrambled*, in the field
`sig_minus_scrambled`. The three names denote one and the same object -- the
sector labels permuted at random across configurations, 2000 shuffles per
seed.
The field name is kept because the JSON was published with that key.

This release was run and checked with Python 3.12.10, NumPy 2.4.3,
SciPy 1.17.1, Matplotlib 3.10.8 and Pillow 12.2.0. Other versions are not
claimed to be tested.

## Data and Code Availability

The seed statistics of the crossover decomposition (Table II) and of both rows
of the gauge-invariant comparison (Table III) are those of the corrected
engine's per-seed values deposited in
`data/null_calibration_results_vecrep_v2.json`. The deterministic analysis
scripts and JSON output files deposited with this record reproduce the
finite-size scaling and the frozen-regime characterization, with the raw
scaling run in `data/stufe2_results.json`, aggregated by
`data/stufe2_analyse.py` into `data/stufe2_analyse.json`.
The Fibonacci values are reproduced by `data/fibonacci_enumeration.py` and its
output `data/fibonacci_enumeration_results.json`, which cover all ten lengths
L = 3-12 (Table VI rows L = 6 and L = 9 were recomputed by this script and
differ from the earlier release; see the paper's Note on optimization); the gate definitions are imported read-only from the companion record
(concept DOI [10.5281/zenodo.19601352](https://doi.org/10.5281/zenodo.19601352)),
whose own `s_delta_sweeps.json` holds the δ sweep of one selected
sequence per length for L = 3, 6, 8, 10 and 12; that file is deposited with
the companion record and is not part of the present package. The δ
sweep behind the fixed- and adapted-settings figures of the Fibonacci section
(14% and 99.5% of the 200 sampled values, and the range up to 2.72) is held in
`data/sector_range_fixed_settings.json`, first deposited with version 2.0 of
this record ([10.5281/zenodo.19656928](https://doi.org/10.5281/zenodo.19656928))
and included here; its Horodecki column is reproducible from the deposited
concurrence column through |S|max = 2*sqrt(1+C^2), while the fixed-settings
angles come from a numerical search whose generating script is not deposited,
so those entries record what the deposited run yielded rather than what the
method yields on a re-run. The per-seed Table I entries and
the historical model rows of Table VII have no deposited generating script (see
the paper's Data and Code Availability section for the full scope). All files are archived on Zenodo
under the concept DOI [10.5281/zenodo.19600752](https://doi.org/10.5281/zenodo.19600752)
(which always resolves to the latest version); see License below.

## License
- Paper, figures, and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE`
- Source code (`*.py`): [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) — see `LICENSE-CODE`
