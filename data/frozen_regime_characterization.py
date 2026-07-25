#!/usr/bin/env python3
"""
Frozen-regime characterization at beta >= 2.5 (factual, no table values).

Long runs with the deposited engines (imported as building blocks, no engine
modification -- both engines cold-start by construction) to characterize the
frozen regime that Table II's beta >= 2.5 rows sit in:
  - number of topological-sector transitions per run
  - mean anyon number
  - I(sector; S) with permutation sigma (same conventions as the deposited
    crossover_decomposition_validator.py: label (Wx, Wy, min(n//2,10)), 20-bin
    S discretization, per-run best-CHSH settings, sigma over shuffle null)
  - the conditional-variance ratio Var(S|sector)/Var(S), defined only where
    Var(S) > 0: on the 40000-measurement runs it stays close to 1, i.e. S is
    NOT a near-deterministic function of the sector label under long sampling;
    at the Table II run scale (v05_table_scale block) Var(S) is exactly zero
    and the ratio is undefined (null).

Full parameter set: L=16, beta in {2.5, 3.0}, n_therm=2000, n_meas=40000
(10x the production runs), n_shuffle=500, seeds {42, 10, 11, 12, 13}.
Engines: BOTH deposited engines at BOTH beta values -- sim/z2_sim_v05.py (the
anchor engine, whose branch carries the paper's Table II values) and
sim/z2_vec.py (the Boltzmann-branch engine).  Engine and branch are reported
with every number; values are never transferred between engines.  Transition
counts are per the stated number of measured sweeps of each block.
An additional block (v05_table_scale) runs the anchor engine at beta=3.0 with
n_meas=4000 -- the Table II protocol scale -- and documents why that
decomposition row is identically zero: the sector label keeps fluctuating
(several Wilson transitions per run, three to four sectors visited), but the
sampled CHSH value is exactly constant (one distinct S value, zero variance),
so any mutual information with it vanishes identically.
NOTE on reading the output: the "parameters" block (n_meas=40000,
transition_counts_are_per="40000 measured sweeps") describes the "v05" and
"vec" blocks only.  The v05_table_scale block is run at n_meas=4000, so its
transition counts are per 4000 measured sweeps.
Output: frozen_regime_characterization_results.json.  NumPy only.
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
import z2_vec as vec
import z2_sim_v05 as v05

L = 16
N_THERM = 2000
N_MEAS = 40000
N_SHUFFLE = 500
SEEDS = [42, 10, 11, 12, 13]
N_BINS = 20


def mi_estimator(secs, sb):
    N = len(secs)
    joint = defaultdict(int); sc = defaultdict(int); sbc = defaultdict(int)
    for s, b in zip(secs, sb):
        joint[(s, b)] += 1; sc[s] += 1; sbc[b] += 1
    val = 0.0
    for (s, b), cnt in joint.items():
        pj = cnt / N; ps = sc[s] / N; pb = sbc[b] / N
        if pj > 0 and ps > 0 and pb > 0:
            val += pj * np.log2(pj / (ps * pb))
    return val


def analyze(data, seed):
    Wx = [r['Wx'] for r in data]; Wy = [r['Wy'] for r in data]
    sec2 = list(zip(Wx, Wy))
    transitions = sum(1 for i in range(1, len(sec2)) if sec2[i] != sec2[i - 1])
    n_sectors = len(set(sec2))
    anyons = float(np.mean([r['n_anyons'] for r in data]))
    C = np.array([[np.mean([r['A'][i] * r['B'][j] for r in data]) for j in range(4)] for i in range(4)])
    best, combo = -1e9, (0, 1, 0, 1)
    for a1 in range(4):
        for a2 in range(4):
            if a1 == a2: continue
            for b1 in range(4):
                for b2 in range(4):
                    if b1 == b2: continue
                    S = C[a1, b1] - C[a1, b2] + C[a2, b1] + C[a2, b2]
                    if abs(S) > best: best, combo = abs(S), (a1, a2, b1, b2)
    a1, a2, b1, b2 = combo
    S_vals = np.array([r['A'][a1] * r['B'][b1] - r['A'][a1] * r['B'][b2]
                       + r['A'][a2] * r['B'][b1] + r['A'][a2] * r['B'][b2] for r in data], float)
    labels = [(r['Wx'], r['Wy'], min(r['n_anyons'] // 2, 10)) for r in data]
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, N_BINS + 1)
    S_binned = np.digitize(S_vals, edges)
    rng = np.random.default_rng(seed)
    I_real = mi_estimator(labels, S_binned)
    I_sh = np.array([mi_estimator([labels[i] for i in rng.permutation(len(labels))], S_binned)
                     for _ in range(N_SHUFFLE)])
    sigma = float((I_real - I_sh.mean()) / I_sh.std()) if I_sh.std() > 0 else 0.0
    var_tot = float(S_vals.var())
    if var_tot > 0:
        groups = defaultdict(list)
        for lab, s in zip(labels, S_vals): groups[lab].append(s)
        var_cond = sum(len(v) * np.var(v) for v in groups.values()) / len(S_vals)
        ratio = float(var_cond / var_tot)
    else:
        ratio = None   # S selbst konstant -> Ratio undefiniert
    return {"transitions": transitions, "n_sectors_visited": n_sectors,
            "anyons_mean": round(anyons, 3), "I": round(float(I_real), 6),
            "sigma": round(sigma, 2),
            "var_S_total": round(var_tot, 6),
            "n_distinct_S_values": int(len(set(np.round(S_vals, 10)))),
            "var_ratio_cond_over_total": (round(ratio, 6) if ratio is not None else None)}


def main():
    results = {"parameters": {"L": L, "n_therm": N_THERM, "n_meas": N_MEAS,
                              "n_shuffle": N_SHUFFLE, "seeds": SEEDS, "n_bins": N_BINS,
                              "transition_counts_are_per": "40000 measured sweeps"},
               "v05": {}, "vec": {}}
    for beta in (2.5, 3.0):
        cell = {}
        for seed in SEEDS:
            t0 = time.time()
            data = v05.run_sim(L=L, beta=beta, n_therm=N_THERM, n_meas=N_MEAS, seed=seed, verbose=False)
            cell[str(seed)] = analyze(data, seed)
            r = cell[str(seed)]
            print(f"  [v05] beta={beta} seed={seed}  {time.time()-t0:5.1f}s  trans={r['transitions']} "
                  f"sect={r['n_sectors_visited']} anyons={r['anyons_mean']} I={r['I']:.4f} "
                  f"({r['sigma']}s.) varratio={r['var_ratio_cond_over_total']}", flush=True)
        results["v05"][str(beta)] = cell
    for beta in (2.5, 3.0):
        cell = {}
        for seed in SEEDS:
            t0 = time.time()
            data = vec.run_sim_vec(L=L, beta=beta, n_therm=N_THERM, n_meas=N_MEAS, seed=seed)
            cell[str(seed)] = analyze(data, seed)
            r = cell[str(seed)]
            print(f"  [vec] beta={beta} seed={seed}  {time.time()-t0:5.1f}s  trans={r['transitions']} "
                  f"sect={r['n_sectors_visited']} anyons={r['anyons_mean']} I={r['I']:.4f} "
                  f"({r['sigma']}s.) varratio={r['var_ratio_cond_over_total']}", flush=True)
        results["vec"][str(beta)] = cell
    # Table-run scale (anchor engine, 4000 measured sweeps = the Table II protocol):
    # documents WHY the beta=3.0 decomposition row is identically zero -- the sector
    # label keeps fluctuating, but the sampled CHSH value is exactly constant
    # (n_distinct_S_values = 1, var_S_total = 0), so any MI with it vanishes.
    results["v05_table_scale"] = {}
    cell = {}
    for seed in SEEDS:
        t0 = time.time()
        data = v05.run_sim(L=L, beta=3.0, n_therm=N_THERM, n_meas=4000, seed=seed, verbose=False)
        cell[str(seed)] = analyze(data, seed)
        r = cell[str(seed)]
        print(f"  [v05@4000] beta=3.0 seed={seed}  {time.time()-t0:5.1f}s  trans={r['transitions']} "
              f"sect={r['n_sectors_visited']} anyons={r['anyons_mean']} nS={r['n_distinct_S_values']} "
              f"varS={r['var_S_total']}", flush=True)
    results["v05_table_scale"]["3.0"] = cell
    out = os.path.join(HERE, "frozen_regime_characterization_results.json")
    io.open(out, "w", encoding="utf-8", newline="\n").write(json.dumps(results, indent=2) + "\n")
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
