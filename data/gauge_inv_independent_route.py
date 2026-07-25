#!/usr/bin/env python3
"""
Independent second-route recompute of the gauge-inv plaquette-strip observable
(Paper 1a re-anchor question). Written from a written physics specification WITHOUT importing or copying the
reference-route script (gauge_inv_plaquette_strip_recompute.py) or the engine
(z2_sim_v05.py). Independent choices made on purpose:
  - MC sweep touches horizontal+vertical edge at each vertex interleaved (not two
    separate full passes), different traversal order than the reference engine.
  - Plaquette array built with explicit modular-shift indexing, not np.roll.
  - Path-product loops written from scratch against the verbal specification.
Same physics (H = -sum B_p, Metropolis with beta), same L/beta/n_therm/n_meas/seeds/
detectors/path_len, same MI+CHSH+permutation-test formulas (so a mismatch is
attributable to a real discrepancy, not a different statistical convention).
"""
import numpy as np
from collections import defaultdict
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))

L = 16
BETA = 2.0
N_THERM = 2000
N_MEAS = 4000
SEEDS = [42] + list(range(10, 30))  # 21 seeds, pre-declared set (pre-declared): 42 + 10..29
DET_A = (L // 4, L // 4)      # (4,4)
DET_B = (3 * L // 4, 3 * L // 4)  # (12,12)
PATH_LEN = L // 4              # 4


class GaugeConfig:
    """Z2 gauge field on L x L torus. h[y,x] = edge (y,x)->(y,x+1); v[y,x] = edge (y,x)->(y+1,x)."""

    def __init__(self, L, seed):
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.h = np.ones((L, L), dtype=np.int64)
        self.v = np.ones((L, L), dtype=np.int64)

    def plaquette_array(self):
        # B[y,x] = h[y,x] * v[y,(x+1)%L] * h[(y+1)%L,x] * v[y,x] -- explicit shift, no np.roll
        L = self.L
        xp1 = np.array([(x + 1) % L for x in range(L)])
        yp1 = np.array([(y + 1) % L for y in range(L)])
        B = self.h * self.v[:, xp1] * self.h[yp1, :] * self.v
        return B

    def sweep(self):
        """Interleaved Metropolis sweep: at each vertex, try flipping h then v edge."""
        L = self.L
        beta = BETA
        h, v, rng = self.h, self.v, self.rng
        for y in range(L):
            ym = (y - 1) % L
            yp = (y + 1) % L
            for x in range(L):
                xp = (x + 1) % L
                xm = (x - 1) % L
                # --- horizontal edge h[y,x]: touches plaquette (y,x) and (ym,x) ---
                bp_here = h[y, x] * v[y, xp] * h[yp, x] * v[y, x]
                bp_above = h[ym, x] * v[ym, xp] * h[y, x] * v[ym, x]
                dE = 2 * (bp_here + bp_above)
                if dE <= 0 or rng.random() < np.exp(-beta * dE):
                    h[y, x] *= -1
                # --- vertical edge v[y,x]: touches plaquette (y,x) and (y,xm) ---
                bp_here = h[y, x] * v[y, xp] * h[yp, x] * v[y, x]
                bp_left = h[y, xm] * v[y, x] * h[yp, xm] * v[y, xm]
                dE = 2 * (bp_here + bp_left)
                if dE <= 0 or rng.random() < np.exp(-beta * dE):
                    v[y, x] *= -1

    def sector(self):
        return int(np.prod(self.h[0, :])), int(np.prod(self.v[:, 0]))

    def n_anyons(self, B=None):
        if B is None:
            B = self.plaquette_array()
        return int(np.sum(B == -1))


DELTAS = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}  # right, up, left, down
RECT_CORNERS = [(0, 1), (1, 2), (2, 3), (3, 0)]
N_DIRECTIONS = 8


def straight_path_product(B, y, x, d, length):
    L = B.shape[0]
    cy, cx = y, x
    prod = 1
    for _ in range(length):
        prod *= int(B[cy, cx])
        dy, dx = DELTAS[d]
        cy, cx = (cy + dy) % L, (cx + dx) % L
    return prod


def rect_path_product(B, y, x, corner, length):
    """L-shaped corner path: leg1 = length//2 steps (incl. origin) on d1, remaining on d2."""
    L = B.shape[0]
    d1, d2 = RECT_CORNERS[corner]
    leg1 = length // 2
    cy, cx = y, x
    prod = int(B[cy, cx])
    for i in range(length - 1):
        d = d1 if i < leg1 - 1 else d2
        dy, dx = DELTAS[d]
        cy, cx = (cy + dy) % L, (cx + dx) % L
        prod *= int(B[cy, cx])
    return prod


def measure_all_directions(B, y, x, length):
    out = {}
    for d in range(4):
        out[d] = straight_path_product(B, y, x, d, length)
    for c in range(4):
        out[4 + c] = rect_path_product(B, y, x, c, length)
    return out


def correlation_matrix(data, n_dir=8):
    C = np.zeros((n_dir, n_dir))
    for i in range(n_dir):
        for j in range(n_dir):
            C[i, j] = np.mean([r['A'][i] * r['B'][j] for r in data])
    return C


def best_chsh(C):
    n_dir = C.shape[0]
    best = 0.0
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


def chsh_series(data, a1, a2, b1, b2):
    return np.array([
        r['A'][a1] * r['B'][b1] - r['A'][a1] * r['B'][b2] +
        r['A'][a2] * r['B'][b1] + r['A'][a2] * r['B'][b2]
        for r in data
    ])


def mutual_info(data, a1, a2, b1, b2, n_shuffle=2000, seed=42):
    rng = np.random.default_rng(seed)
    sectors = [(r['Wx'], r['Wy'], min(r['n_any'] // 2, 10)) for r in data]
    S_vals = chsh_series(data, a1, a2, b1, b2)
    n_bins = 20
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, n_bins + 1)
    S_bin = np.digitize(S_vals, edges)

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

    I_real = mi(sectors, S_bin)
    I_sh = np.array([
        mi([sectors[i] for i in rng.permutation(len(sectors))], S_bin)
        for _ in range(n_shuffle)
    ])
    sigma = (I_real - I_sh.mean()) / I_sh.std() if I_sh.std() > 0 else float('nan')
    return I_real, sigma


def run_seed(seed):
    t0 = time.time()
    cfg = GaugeConfig(L, seed)
    for _ in range(N_THERM):
        cfg.sweep()
    t_therm = time.time() - t0

    data = []
    t0 = time.time()
    for _ in range(N_MEAS):
        cfg.sweep()
        B = cfg.plaquette_array()
        Wx, Wy = cfg.sector()
        n_any = cfg.n_anyons(B)
        A = measure_all_directions(B, DET_A[0], DET_A[1], PATH_LEN)
        Bd = measure_all_directions(B, DET_B[0], DET_B[1], PATH_LEN)
        data.append({'Wx': Wx, 'Wy': Wy, 'n_any': n_any, 'A': A, 'B': Bd})
    t_meas = time.time() - t0

    C = correlation_matrix(data, n_dir=N_DIRECTIONS)
    S_best, combo = best_chsh(C)
    I_real, sigma = mutual_info(data, *combo, n_shuffle=2000, seed=42)
    print(f"seed={seed}: I={I_real:.4f} sigma={sigma:.2f} S_best={S_best:.3f} combo={combo} "
          f"(therm {t_therm:.1f}s, meas {t_meas:.1f}s)", flush=True)
    return {'seed': seed, 'I': I_real, 'sigma': sigma, 'S_best': S_best, 'combo': combo}


def main():
    results = [run_seed(s) for s in SEEDS]
    Is = np.array([r['I'] for r in results])
    sigmas = np.array([r['sigma'] for r in results])
    n_gt3 = int(np.sum(sigmas > 3))
    I_mean = float(Is.mean())
    I_sem = float(Is.std(ddof=1) / np.sqrt(len(Is)))
    sigma_mean = float(sigmas.mean())
    sigma_sem = float(sigmas.std(ddof=1) / np.sqrt(len(sigmas)))

    print("\n=== INDEPENDENT SECOND-ROUTE SUMMARY ===")
    print(f"I = {I_mean:.4f} +/- {I_sem:.4f} bits, sigma = {sigma_mean:.1f} +/- {sigma_sem:.1f}, "
          f"{n_gt3}/{len(SEEDS)} seeds >3sigma")
    print(f"Published (v2.3): I = 0.018 +/- 0.011, sigma = 14.2 +/- 11.4, 21/21 >3sigma")

    out = {
        'I_mean': I_mean, 'I_sem': I_sem,
        'sigma_mean': sigma_mean, 'sigma_sem': sigma_sem,
        'seeds_gt_3sigma': f"{n_gt3}/{len(SEEDS)}",
        'per_seed': results,
    }
    out_path = os.path.join(HERE, 'gauge_inv_independent_route_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == '__main__':
    main()
