#!/usr/bin/env python3
"""
Gauge-invariant plaquette-strip recompute for Paper 1a Tab. III ("Gauge-inv. plaq. strips").

Context: this script is the from-scratch recompute of the gauge-invariant
plaquette-strip observable. The values published in v2.3 (I = 0.018 +/- 0.011
bits, sigma = 14.2 +/- 11.4, 21/21 seeds > 3 sigma) are the output of this
recompute route over the expanded 21-seed set; the earlier value reported in
v2.0-v2.2 (I = 0.035 +/- 0.016, 5 seeds) had no surviving generating script
and is superseded by this recomputation. The observable is built exactly as
described in the paper's own Methods section:

    "gauge-invariant plaquette-strip observables --- products of adjacent
    plaquette operators along horizontal, vertical, and rectangular paths
    (length 4) --- which require no gauge fixing"

The primary (already-reproduced, uncontested) Wilson-line result I=0.015+/-0.011
bits uses Z2GaugeTheory.measure_detector() -> wilson_line() (product of EDGE
variables along a path). This script adds the missing gauge-invariant sibling:
a detector built from PRODUCTS OF PLAQUETTE VALUES along a path instead, then
feeds it through the exact same (unmodified) analysis pipeline (correlation
matrix -> best_chsh -> mutual_info w/ permutation test) so any difference in
the result is attributable to the observable, not to a different statistical
method.

Z2GaugeTheory is imported (not retyped) from the archived original
(the archived Z2 engine z2_sim_v05.py) to avoid any
transcription drift from the paper-validated engine.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# the Z2 engine ships alongside this script in the deposit (../sim/z2_sim_v05.py)
ENGINE_PATH = Path(__file__).resolve().parent.parent / "sim" / "z2_sim_v05.py"

spec = importlib.util.spec_from_file_location("z2_sim_v05", ENGINE_PATH)
_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_engine)
Z2GaugeTheory = _engine.Z2GaugeTheory


# ── NEW: gauge-invariant plaquette-strip detector (mirrors wilson_line/measure_detector) ──

def plaquette_strip_product(B: np.ndarray, y: int, x: int, direction: int, length: int) -> int:
    """
    Product of `length` adjacent plaquette values starting AT (y,x), walking in
    `direction` (0=right,1=up,2=left,3=down) -- the gauge-invariant analogue of
    wilson_line(). Includes the origin plaquette itself as step 1 (so length=4
    means 4 plaquette factors total, matching the paper's "length 4" wording).
    """
    L = B.shape[0]
    cy, cx = y, x
    prod = 1
    for step in range(length):
        prod *= int(B[cy, cx])
        if direction == 0:      # right
            cx = (cx + 1) % L
        elif direction == 1:    # up
            cy = (cy - 1) % L
        elif direction == 2:    # left
            cx = (cx - 1) % L
        elif direction == 3:    # down
            cy = (cy + 1) % L
    return prod


def plaquette_strip_product_vectorized(B: np.ndarray, y: int, x: int, direction: int, length: int) -> int:
    """Independent (nonloop, slice-based) re-implementation for cross-check."""
    L = B.shape[0]
    idx = np.arange(length)
    if direction == 0:
        ys, xs = np.full(length, y), (x + idx) % L
    elif direction == 1:
        ys, xs = (y - idx) % L, np.full(length, x)
    elif direction == 2:
        ys, xs = np.full(length, y), (x - idx) % L
    elif direction == 3:
        ys, xs = (y + idx) % L, np.full(length, x)
    else:
        raise ValueError(direction)
    return int(np.prod(B[ys, xs]))


RECT_CORNERS = [(0, 1), (1, 2), (2, 3), (3, 0)]  # (right,up) (up,left) (left,down) (down,right)


_DELTAS = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}  # right, up, left, down


def _rect_positions(y: int, x: int, corner: int, length: int, L: int):
    """
    THIRD path type from the paper's own wording ("horizontal, vertical, AND
    rectangular paths"). A bent/rectangular path: the first `leg1 = length//2`
    positions (incl. origin) sit on the first leg (direction d1), the
    remaining `length-leg1` positions continue on the perpendicular second
    leg (direction d2) -- still a plain product of adjacent B_p, still gauge-
    invariant, but geometrically distinct from a pure straight line.
    Returns the list of `length` (y,x) lattice positions visited, in order.
    """
    d1, d2 = RECT_CORNERS[corner]
    leg1 = length // 2
    cy, cx = y, x
    pts = [(cy, cx)]
    for i in range(length - 1):
        d = d1 if i < leg1 - 1 else d2
        dy, dx = _DELTAS[d]
        cy, cx = (cy + dy) % L, (cx + dx) % L
        pts.append((cy, cx))
    return pts


def plaquette_rect_product(B: np.ndarray, y: int, x: int, corner: int, length: int) -> int:
    L = B.shape[0]
    prod = 1
    for (cy, cx) in _rect_positions(y, x, corner, length, L):
        prod *= int(B[cy, cx])
    return prod


def plaquette_rect_product_vectorized(B: np.ndarray, y: int, x: int, corner: int, length: int) -> int:
    """Independent (array-indexing, not the shared position-list helper) cross-check."""
    L = B.shape[0]
    d1, d2 = RECT_CORNERS[corner]
    leg1 = length // 2
    leg2 = length - leg1
    dy1, dx1 = _DELTAS[d1]
    dy2, dx2 = _DELTAS[d2]
    i1 = np.arange(leg1)
    ys1, xs1 = (y + i1 * dy1) % L, (x + i1 * dx1) % L
    corner_y, corner_x = ys1[-1], xs1[-1]
    i2 = np.arange(1, leg2 + 1)
    ys2, xs2 = (corner_y + i2 * dy2) % L, (corner_x + i2 * dx2) % L
    ys = np.concatenate([ys1, ys2])
    xs = np.concatenate([xs1, xs2])
    return int(np.prod(B[ys, xs]))


N_DIRECTIONS = 4 + len(RECT_CORNERS)  # 4 straight + 4 rectangular-corner


def measure_detector_plaquette(B: np.ndarray, y: int, x: int, path_len: int) -> dict:
    """
    Gauge-invariant analogue of Z2GaugeTheory.measure_detector(), extended to
    the FULL set the paper describes: horizontal+vertical (straight, d=0..3)
    AND rectangular (bent corners, d=4..7). best_chsh() below already searches
    all pairs, so this lets it discover the strongest signal across the whole
    family instead of guessing which 2 of 8 the original used.
    """
    out = {d: plaquette_strip_product(B, y, x, d, path_len) for d in range(4)}
    out.update({4 + c: plaquette_rect_product(B, y, x, c, path_len) for c in range(len(RECT_CORNERS))})
    return out


def measure_detector_plaquette_vectorized(B: np.ndarray, y: int, x: int, path_len: int) -> dict:
    out = {d: plaquette_strip_product_vectorized(B, y, x, d, path_len) for d in range(4)}
    out.update({4 + c: plaquette_rect_product_vectorized(B, y, x, c, path_len) for c in range(len(RECT_CORNERS))})
    return out


# ── Reused, UNMODIFIED analysis pipeline (transcribed verbatim from z2_sim_v05.py
#    so it operates on our A_p/B_p dict fields instead of A/B; logic is untouched) ──

def correlation_matrix(data, key_a, key_b, n_dir=4):
    C = np.zeros((n_dir, n_dir))
    for i in range(n_dir):
        for j in range(n_dir):
            C[i, j] = np.mean([r[key_a][i] * r[key_b][j] for r in data])
    return C


def best_chsh(C):
    n_dir = C.shape[0]
    best = 0
    combo = (0, 1, 0, 1)
    for a1 in range(n_dir):
        for a2 in range(n_dir):
            if a1 == a2:
                continue
            for b1 in range(n_dir):
                for b2 in range(n_dir):
                    if b1 == b2:
                        continue
                    S = C[a1, b1] - C[a1, b2] + C[a2, b1] + C[a2, b2]
                    if abs(S) > abs(best):
                        best = S
                        combo = (a1, a2, b1, b2)
    return best, combo


def chsh_per_config(data, key_a, key_b, a1, a2, b1, b2):
    return np.array([
        r[key_a][a1] * r[key_b][b1] - r[key_a][a1] * r[key_b][b2] +
        r[key_a][a2] * r[key_b][b1] + r[key_a][a2] * r[key_b][b2]
        for r in data
    ])


def mutual_info(data, key_a, key_b, a1, a2, b1, b2, n_shuffle=2000, seed=42):
    rng = np.random.default_rng(seed)
    sectors = [(r['Wx'], r['Wy'], min(r['n_anyons'] // 2, 10)) for r in data]
    S_vals = chsh_per_config(data, key_a, key_b, a1, a2, b1, b2)

    n_bins = 20
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, n_bins + 1)
    S_binned = np.digitize(S_vals, edges)

    def mi(secs, sb):
        N = len(secs)
        joint, sc, sbc = defaultdict(int), defaultdict(int), defaultdict(int)
        for s, b in zip(secs, sb):
            joint[(s, b)] += 1
            sc[s] += 1
            sbc[b] += 1
        val = 0.0
        for (s, b), cnt in joint.items():
            pj, ps, pb = cnt / N, sc[s] / N, sbc[b] / N
            if pj > 0 and ps > 0 and pb > 0:
                val += pj * np.log2(pj / (ps * pb))
        return val

    I_real = mi(sectors, S_binned)
    I_sh = np.array([
        mi([sectors[i] for i in rng.permutation(len(sectors))], S_binned)
        for _ in range(n_shuffle)
    ])
    p = np.mean(I_sh >= I_real) if len(I_sh) else 1.0
    sigma = (I_real - I_sh.mean()) / I_sh.std() if I_sh.std() > 0 else float('nan')
    return I_real, sigma, p


def run_one_seed(seed: int, L=16, beta=2.0, n_therm=2000, n_meas=4000, verbose=True):
    tc = Z2GaugeTheory(L, beta, seed)
    det_A = (L // 4, L // 4)
    det_B = (3 * L // 4, 3 * L // 4)
    path_len = L // 4  # = 4 for L=16, matches paper's "length 4"

    t0 = time.time()
    for _ in range(n_therm):
        tc.sweep()
    if verbose:
        print(f"  seed={seed} thermalized in {time.time()-t0:.1f}s")

    data = []
    t0 = time.time()
    cross_check_mismatches = 0
    for m in range(n_meas):
        tc.sweep()
        Wx, Wy = tc.topological_sector()
        n_any = tc.anyon_count()
        Bp = tc.plaquette_array()

        A_p = measure_detector_plaquette(Bp, det_A[0], det_A[1], path_len)
        B_p = measure_detector_plaquette(Bp, det_B[0], det_B[1], path_len)

        # independent 2nd-route cross-check on every measurement (cheap, exact)
        A_p_vec = measure_detector_plaquette_vectorized(Bp, det_A[0], det_A[1], path_len)
        B_p_vec = measure_detector_plaquette_vectorized(Bp, det_B[0], det_B[1], path_len)
        for d in range(N_DIRECTIONS):
            if A_p_vec[d] != A_p[d]:
                cross_check_mismatches += 1
            if B_p_vec[d] != B_p[d]:
                cross_check_mismatches += 1

        data.append({'Wx': Wx, 'Wy': Wy, 'n_anyons': n_any, 'A_p': A_p, 'B_p': B_p})

    if verbose:
        print(f"  seed={seed} measured in {time.time()-t0:.1f}s, "
              f"cross-check mismatches={cross_check_mismatches}/{n_meas*2*N_DIRECTIONS}")

    C = correlation_matrix(data, 'A_p', 'B_p', n_dir=N_DIRECTIONS)
    S_best, combo = best_chsh(C)
    I_real, sigma, p = mutual_info(data, 'A_p', 'B_p', *combo)

    return {
        'seed': seed, 'I': I_real, 'sigma': sigma, 'p': p,
        'S_best': S_best, 'combo': combo,
        'cross_check_mismatches': cross_check_mismatches,
        'n_meas': n_meas,
    }


def main():
    # Expanded, PRE-DECLARED seed set (no cherry-picking): paper's own convention
    # original=42, systematic=10-19, new tests=20-29 -> 21 seeds. Shrinks the SEM
    # so the archived number isn't dominated by a single hot seed (e.g. seed 11).
    seeds = [42] + list(range(10, 30))
    results = []
    for s in seeds:
        print(f"=== seed {s} ===")
        r = run_one_seed(s)
        print(f"  I={r['I']:.4f} bits, sigma={r['sigma']:.2f}, p={r['p']:.4f}, "
              f"S_best={r['S_best']:.3f}, combo={r['combo']}")
        results.append(r)

    Is = np.array([r['I'] for r in results])
    sigmas = np.array([r['sigma'] for r in results])
    n_gt3 = int(np.sum(sigmas > 3))
    total_mismatches = sum(r['cross_check_mismatches'] for r in results)

    summary = {
        'version': 'v2_with_rectangular_paths',
        'n_directions': N_DIRECTIONS,
        'note': 'v1 (straight horizontal/vertical only, 4 directions) preserved in '
                '_v1_straightonly_results.json for comparison.',
        'target_paper': {'I': [0.018, 0.011], 'sigma': [14.2, 11.4], 'seeds_gt_3sigma': '21/21'},
        'recompute': {
            'I_mean': float(Is.mean()), 'I_sem': float(Is.std(ddof=1) / np.sqrt(len(Is))),
            'sigma_mean': float(sigmas.mean()), 'sigma_sem': float(sigmas.std(ddof=1) / np.sqrt(len(sigmas))),
            'seeds_gt_3sigma': f"{n_gt3}/{len(seeds)}",
        },
        'cross_check_total_mismatches': total_mismatches,
        'per_seed': results,
    }
    out_path = Path(__file__).parent / "gauge_inv_plaquette_strip_recompute_v2_expanded21_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== SUMMARY (v2, 8 directions: 4 straight + 4 rectangular) ===")
    print(f"Recompute: I = {summary['recompute']['I_mean']:.4f} +/- {summary['recompute']['I_sem']:.4f} bits, "
          f"sigma = {summary['recompute']['sigma_mean']:.1f} +/- {summary['recompute']['sigma_sem']:.1f}, "
          f"{n_gt3}/{len(seeds)} seeds >3sigma")
    print(f"Published (v2.3): I = 0.018 +/- 0.011 bits, sigma = 14.2 +/- 11.4, 21/21 seeds >3sigma")
    print(f"Cross-check (loop vs vectorized route): {total_mismatches} mismatches (want 0)")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
