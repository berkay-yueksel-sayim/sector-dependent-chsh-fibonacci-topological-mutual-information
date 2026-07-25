#!/usr/bin/env python3
"""
Independent second-route check of the Table II crossover decomposition.

Uses the deposited SECOND Monte Carlo engine (sim/z2_vec.py -- independently
implemented, vectorized checkerboard Metropolis) together with an independently
written MI estimator (np.unique on encoded label/bin pairs, no dict loops),
so both the sampling engine and the MI arithmetic differ from the validator
route (crossover_decomposition_validator.py, which uses sim/z2_sim_v05.py and
the dict-based estimator formulas).

SCOPE CONTRACT (deliberate, documented): as recorded in sim/z2_vec.py and in
the paper's finite-size-scaling section, the two engines reach different
quasi-equilibria at beta = 2.0 (vec ~ full Boltzmann, more anyons; v05 = the
paper's not-yet-converged low-temperature branch, whose values Tables I--II
report).  This route therefore checks the STRUCTURE of the decomposition --
which component carries the significance at each beta, and the crossover
ordering -- not the absolute I values, which are engine-specific by the
paper's own disclosure.

Deliberately identical conventions: detectors A=(L/4,L/4), B=(3L/4,3L/4),
4 Wilson lines of length L/4, sector label (Wx, Wy, min(n_anyons//2, 10)),
20-bin S discretization, per-run best-CHSH combo, permutation sigma
(I_real - mean(I_shuffled)) / std(I_shuffled).
Full parameter set: L=16, beta in {1.0,1.5,2.0,2.5,3.0}, n_therm=2000,
n_meas=4000, n_shuffle=2000, seeds {42,10,11,12,13}.
Output: crossover_independent_route_results.json.  NumPy only.
"""
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import z2_vec as engine2  # deposited second engine (independent implementation)

L = 16
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
N_THERM = 2000
N_MEAS = 4000
N_SHUFFLE = 2000
SEEDS = [42, 10, 11, 12, 13]
N_BINS = 20


def mi_bits(codes_a, codes_b):
    """Plug-in MI in bits via np.unique -- independent arithmetic route."""
    n = len(codes_a)
    _, ia = np.unique(codes_a, return_inverse=True)
    _, ib = np.unique(codes_b, return_inverse=True)
    pair = ia.astype(np.int64) * (int(ib.max()) + 1) + ib
    _, pc = np.unique(pair, return_counts=True)
    _, ca = np.unique(ia, return_counts=True)
    _, cb = np.unique(ib, return_counts=True)
    Hj = -np.sum((pc / n) * np.log2(pc / n))
    Ha = -np.sum((ca / n) * np.log2(ca / n))
    Hb = -np.sum((cb / n) * np.log2(cb / n))
    return float(Ha + Hb - Hj)


def mi_with_sigma(codes, S_binned, n_shuffle, seed):
    rng = np.random.default_rng(seed)
    I_real = mi_bits(codes, S_binned)
    n = len(codes)
    I_sh = np.empty(n_shuffle)
    for k in range(n_shuffle):
        I_sh[k] = mi_bits(codes[rng.permutation(n)], S_binned)
    sigma = (I_real - I_sh.mean()) / I_sh.std() if I_sh.std() > 0 else 0.0
    return float(I_real), float(sigma)


def run_cell(beta, seed):
    data = engine2.run_sim_vec(L=L, beta=beta, n_therm=N_THERM,
                               n_meas=N_MEAS, seed=seed)
    A = np.array([[r['A'][d] for d in range(4)] for r in data], dtype=np.int8)
    Bm = np.array([[r['B'][d] for d in range(4)] for r in data], dtype=np.int8)
    C = np.array([[np.mean(A[:, i] * Bm[:, j]) for j in range(4)] for i in range(4)])
    best, combo = -1e9, None
    for a1 in range(4):
        for a2 in range(4):
            if a1 == a2: continue
            for b1 in range(4):
                for b2 in range(4):
                    if b1 == b2: continue
                    S = C[a1, b1] - C[a1, b2] + C[a2, b1] + C[a2, b2]
                    if abs(S) > best: best, combo = abs(S), (a1, a2, b1, b2)
    a1, a2, b1, b2 = combo
    S_vals = (A[:, a1] * Bm[:, b1] - A[:, a1] * Bm[:, b2]
              + A[:, a2] * Bm[:, b1] + A[:, a2] * Bm[:, b2]).astype(float)
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, N_BINS + 1)
    S_binned = np.digitize(S_vals, edges)
    Wx = np.array([r['Wx'] for r in data], dtype=np.int64)
    Wy = np.array([r['Wy'] for r in data], dtype=np.int64)
    abin = np.minimum(np.array([r['n_anyons'] for r in data]) // 2, 10).astype(np.int64)
    enc_full = ((Wx + 1) // 2 * 2 + (Wy + 1) // 2) * 16 + abin
    enc_wilson = (Wx + 1) // 2 * 2 + (Wy + 1) // 2
    enc_anyons = abin
    out = {"settings": [int(v) for v in combo], "S_best": float(best)}
    for name, enc in (("full", enc_full), ("wilson", enc_wilson), ("anyons", enc_anyons)):
        I, sig = mi_with_sigma(enc, S_binned, N_SHUFFLE, seed)
        out["I_" + name] = round(I, 6)
        out["sigma_" + name] = round(sig, 2)
    out["anyons_mean"] = round(float(np.mean([r['n_anyons'] for r in data])), 2)
    return out


def main():
    results = {"parameters": {"L": L, "betas": BETAS, "n_therm": N_THERM,
                              "n_meas": N_MEAS, "n_shuffle": N_SHUFFLE,
                              "seeds": SEEDS, "n_bins": N_BINS,
                              "route": "independent (deposited z2_vec engine + np.unique MI)"},
               "per_beta": {}}
    for beta in BETAS:
        cell = {"per_seed": {}}
        for seed in SEEDS:
            t0 = time.time()
            cell["per_seed"][str(seed)] = run_cell(beta, seed)
            r = cell["per_seed"][str(seed)]
            print(f"  [route2] beta={beta} seed={seed}  {time.time()-t0:5.1f}s  "
                  f"I_full={r['I_full']:.4f} ({r['sigma_full']:.1f}s.) "
                  f"I_w={r['I_wilson']:.4f} ({r['sigma_wilson']:.1f}s.) "
                  f"I_a={r['I_anyons']:.4f} ({r['sigma_anyons']:.1f}s.)", flush=True)
        for comp in ("full", "wilson", "anyons"):
            vals = [cell["per_seed"][str(s)]["I_" + comp] for s in SEEDS]
            cell["I_" + comp + "_mean"] = round(float(np.mean(vals)), 6)
        results["per_beta"][str(beta)] = cell
    out = os.path.join(HERE, "crossover_independent_route_results.json")
    io.open(out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(results, indent=2) + "\n")
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
