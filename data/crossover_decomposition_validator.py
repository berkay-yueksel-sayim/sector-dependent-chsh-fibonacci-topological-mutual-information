#!/usr/bin/env python3
"""
Validation script for the Table II information-source crossover decomposition.

Recomputes, from the deposited Monte Carlo engine (sim/z2_sim_v05.py, imported
as a building block -- no engine code is duplicated here), the mutual-information
decomposition of the sector label into its Wilson-loop and anyonic components:

    I_full   = I(sector; S)  with sector = (Wx, Wy, anyon_bin)
    I_wilson = I(sector; S)  with sector = (Wx, Wy)
    I_anyons = I(sector; S)  with sector = (anyon_bin,)

where anyon_bin = min(n_anyons // 2, 10), exactly as built inside
z2_sim_v05.mutual_info.  The MI estimator, the 20-bin S-discretization, and the
permutation-test sigma are re-expressed here with the identical formulas
(the estimator inside z2_sim_v05.mutual_info is a nested function and therefore
not importable); the S values, binning edges, and shuffle convention
(rng.permutation of the sector list, sigma = (I_real - mean(I_shuffled)) /
std(I_shuffled)) are identical for all three projections, so the three MI values
per run are computed on the same binned S sequence.

Full parameter set (declared, not implied):
    L = 16 . beta in {1.0, 1.5, 2.0, 2.5, 3.0} . n_therm = 2000 . n_meas = 4000
    n_shuffle = 2000 . seeds = {42, 10, 11, 12, 13}
    (seed 42 = the original discovery-run seed carried by Tables I--II;
     seeds 10-13 = the first four systematic seeds of Table I)
    CHSH settings per run: best_chsh(correlation_matrix(data)) -- the same
    direction-combo optimization used by the deposited engine's main analysis.

Output: crossover_decomposition_results.json (per beta x seed: I_full/I_wilson/
I_anyons with per-component sigma, plus per-beta seed-42 values and seed means).
Deterministic for a fixed parameter set.  NumPy only.  Runtime ~15 min.
"""
import io
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import z2_sim_v05 as engine  # deposited MC engine, used as building block

L = 16
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
N_THERM = 2000
N_MEAS = 4000
N_SHUFFLE = 2000
SEEDS = [42, 10, 11, 12, 13]
N_BINS = 20


def mi_estimator(secs, sb):
    """Plug-in MI in bits -- identical formula to z2_sim_v05.mutual_info.mi."""
    N = len(secs)
    joint = defaultdict(int)
    sc = defaultdict(int)
    sbc = defaultdict(int)
    for s, b in zip(secs, sb):
        joint[(s, b)] += 1
        sc[s] += 1
        sbc[b] += 1
    val = 0.0
    for (s, b), cnt in joint.items():
        pj = cnt / N
        ps = sc[s] / N
        pb = sbc[b] / N
        if pj > 0 and ps > 0 and pb > 0:
            val += pj * np.log2(pj / (ps * pb))
    return val


def mi_with_sigma(labels, S_binned, n_shuffle, seed):
    """I(labels; S_binned) + permutation sigma, same convention as the engine."""
    rng = np.random.default_rng(seed)
    I_real = mi_estimator(labels, S_binned)
    n = len(labels)
    I_sh = np.array([
        mi_estimator([labels[i] for i in rng.permutation(n)], S_binned)
        for _ in range(n_shuffle)
    ])
    sigma = (I_real - I_sh.mean()) / I_sh.std() if I_sh.std() > 0 else 0.0
    return float(I_real), float(sigma)


def decompose(data, seed):
    """The three projections on the identical binned S sequence."""
    C = engine.correlation_matrix(data)
    S_best, (a1, a2, b1, b2) = engine.best_chsh(C)
    S_vals = engine.chsh_per_config(data, a1, a2, b1, b2)
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, N_BINS + 1)
    S_binned = np.digitize(S_vals, edges)

    full = [(r['Wx'], r['Wy'], min(r['n_anyons'] // 2, 10)) for r in data]
    wilson = [(r['Wx'], r['Wy']) for r in data]
    anyons = [(min(r['n_anyons'] // 2, 10),) for r in data]

    out = {"settings": [int(a1), int(a2), int(b1), int(b2)], "S_best": float(S_best)}
    for name, labels in (("full", full), ("wilson", wilson), ("anyons", anyons)):
        I, sig = mi_with_sigma(labels, S_binned, N_SHUFFLE, seed)
        out["I_" + name] = round(I, 6)
        out["sigma_" + name] = round(sig, 2)
    out["anyons_mean"] = round(float(np.mean([r['n_anyons'] for r in data])), 2)
    return out


def main():
    results = {"parameters": {"L": L, "betas": BETAS, "n_therm": N_THERM,
                              "n_meas": N_MEAS, "n_shuffle": N_SHUFFLE,
                              "seeds": SEEDS, "n_bins": N_BINS,
                              "settings_rule": "best_chsh(correlation_matrix(data)) per run"},
               "per_beta": {}}
    for beta in BETAS:
        cell = {"per_seed": {}}
        for seed in SEEDS:
            t0 = time.time()
            data = engine.run_sim(L=L, beta=beta, n_therm=N_THERM,
                                  n_meas=N_MEAS, seed=seed, verbose=False)
            cell["per_seed"][str(seed)] = decompose(data, seed)
            print(f"  beta={beta} seed={seed}  {time.time()-t0:5.1f}s  "
                  f"I_full={cell['per_seed'][str(seed)]['I_full']:.4f} "
                  f"({cell['per_seed'][str(seed)]['sigma_full']:.1f}s.) "
                  f"I_w={cell['per_seed'][str(seed)]['I_wilson']:.4f} "
                  f"({cell['per_seed'][str(seed)]['sigma_wilson']:.1f}s.) "
                  f"I_a={cell['per_seed'][str(seed)]['I_anyons']:.4f} "
                  f"({cell['per_seed'][str(seed)]['sigma_anyons']:.1f}s.)", flush=True)
        for comp in ("full", "wilson", "anyons"):
            vals = [cell["per_seed"][str(s)]["I_" + comp] for s in SEEDS]
            cell["I_" + comp + "_seed42"] = cell["per_seed"]["42"]["I_" + comp]
            cell["sigma_" + comp + "_seed42"] = cell["per_seed"]["42"]["sigma_" + comp]
            cell["I_" + comp + "_mean"] = round(float(np.mean(vals)), 6)
        cell["anyons_mean_seed42"] = cell["per_seed"]["42"]["anyons_mean"]
        results["per_beta"][str(beta)] = cell
    out = os.path.join(HERE, "crossover_decomposition_results.json")
    io.open(out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(results, indent=2) + "\n")
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
