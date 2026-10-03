#!/usr/bin/env python3
"""
===============================================================
Z2 GAUGE THEORY — TORIC CODE SIMULATION v0.5
Z2 Lattice Gauge Theory on LxL Torus
Methodology: Same as v0.1–v0.4 + I1 analysis, new model.
===============================================================

Model: H = -K Sigma_p B_p, where B_p = Pi_{e in p} sigma_e for sigma_e  in  {+1,-1}
       on edges of a square lattice with periodic boundary conditions.

Topological sectors: Noncontractible Wilson loops W_x, W_y  in  {+1,-1}
Anyons: Plaquettes with B_p = -1 (come in pairs)
Measurements: Wilson lines from detectors in different directions (gauge-fixed)
"""

import numpy as np
from collections import defaultdict
import time

# ===================================================
# Z2 GAUGE THEORY ENGINE
# ===================================================

class Z2GaugeTheory:
    """
    Z2 lattice gauge theory on LxL torus.
    
    Edge layout:
      sigma_h[y, x] = horizontal edge from (y,x) to (y, x+1 mod L)
      sigma_v[y, x] = vertical edge from (y,x) to (y+1 mod L, x)
    
    Plaquette at (y,x):
      B_p = sigma_h[y,x] * sigma_v[y, (x+1)%L] * sigma_h[(y+1)%L, x] * sigma_v[y, x]
      (right, down-right, bottom, down-left — counterclockwise)
    """

    def __init__(self, L, beta, seed=42):
        self.L = L
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        # Start in ordered state (all +1, ground state)
        self.sh = np.ones((L, L), dtype=np.int8)
        self.sv = np.ones((L, L), dtype=np.int8)

    def plaquette_array(self):
        """Vectorized: all plaquette values B_p[y,x]."""
        L = self.L
        return (self.sh * np.roll(self.sv, -1, axis=1) *
                np.roll(self.sh, -1, axis=0) * self.sv)

    def energy(self):
        return -int(np.sum(self.plaquette_array()))

    def sweep(self):
        """One Metropolis sweep over all edges."""
        L = self.L
        # --- horizontal edges ---
        for y in range(L):
            for x in range(L):
                # Plaquettes touching sh[y,x]: (y,x) and ((y-1)%L, x)
                ym = (y - 1) % L
                bp1 = (self.sh[y,x] * self.sv[y,(x+1)%L] *
                       self.sh[(y+1)%L,x] * self.sv[y,x])
                bp2 = (self.sh[ym,x] * self.sv[ym,(x+1)%L] *
                       self.sh[y,x] * self.sv[ym,x])
                dE = 2 * (bp1 + bp2)  # flipping sh -> each B_p flips sign, DeltaE = 2*B_old per plaq
                if dE <= 0 or self.rng.random() < np.exp(-self.beta * dE):
                    self.sh[y, x] *= -1

        # --- vertical edges ---
        for y in range(L):
            for x in range(L):
                xm = (x - 1) % L
                bp1 = (self.sh[y,x] * self.sv[y,(x+1)%L] *
                       self.sh[(y+1)%L,x] * self.sv[y,x])
                bp2 = (self.sh[y,xm] * self.sv[y,x] *
                       self.sh[(y+1)%L,xm] * self.sv[y,xm])
                dE = 2 * (bp1 + bp2)
                if dE <= 0 or self.rng.random() < np.exp(-self.beta * dE):
                    self.sv[y, x] *= -1

    def wilson_x(self, y=0):
        """Noncontractible Wilson loop along x-direction at row y."""
        return int(np.prod(self.sh[y, :]))

    def wilson_y(self, x=0):
        """Noncontractible Wilson loop along y-direction at column x."""
        return int(np.prod(self.sv[:, x]))

    def topological_sector(self):
        return (self.wilson_x(0), self.wilson_y(0))

    def anyon_count(self):
        """Number of plaquettes with B_p = -1."""
        return int(np.sum(self.plaquette_array() == -1))

    def anyon_positions(self):
        B = self.plaquette_array()
        return list(zip(*np.where(B == -1)))

    # --- Gauge fixing ---
    def gauge_fix_tree(self):
        """
        Fix maximal-tree gauge. After this, all edges in the spanning tree = +1.
        Spanning tree: all vertical edges + first row of horizontal edges,
        minus the edge that closes the tree.
        
        Returns copies (does not modify self for MC).
        """
        L = self.L
        sh = self.sh.copy()
        sv = self.sv.copy()

        # Fix gauge vertex by vertex using vertical edges as tree
        # For each column x, walk down rows fixing vertical edges
        for x in range(L):
            for y in range(L - 1):
                if sv[y, x] == -1:
                    # Gauge transform at vertex (y+1, x): flip all touching edges
                    self._gauge_transform(sh, sv, (y + 1) % L, x)

        # Now fix horizontal edges along row 0 (except last one for topology)
        for x in range(L - 1):
            if sh[0, x] == -1:
                self._gauge_transform(sh, sv, 0, (x + 1) % L)

        return sh, sv

    @staticmethod
    def _gauge_transform(sh, sv, y, x):
        """Flip all edges touching vertex (y,x)."""
        L = sh.shape[0]
        sh[y, x] *= -1                    # right
        sh[y, (x - 1) % L] *= -1          # left
        sv[y, x] *= -1                     # down
        sv[(y - 1) % L, x] *= -1          # up

    # --- Measurements (on gauge-fixed config) ---
    def wilson_line(self, sh, sv, y, x, direction, length):
        """
        Product of edges along a path from (y,x).
        Directions: 0=right, 1=up, 2=left, 3=down
        """
        L = self.L
        prod = 1
        cy, cx = y, x
        for _ in range(length):
            if direction == 0:  # right
                prod *= sh[cy, cx]
                cx = (cx + 1) % L
            elif direction == 1:  # up
                cy = (cy - 1) % L
                prod *= sv[cy, cx]
            elif direction == 2:  # left
                cx = (cx - 1) % L
                prod *= sh[cy, cx]
            elif direction == 3:  # down
                prod *= sv[cy, cx]
                cy = (cy + 1) % L
        return prod

    def measure_detector(self, sh, sv, y, x, path_len):
        """
        Measure 4 Wilson lines from detector in 4 directions.
        Returns dict {direction: +/-1}.
        """
        return {d: self.wilson_line(sh, sv, y, x, d, path_len)
                for d in range(4)}

    # --- Gauge-invariant alternative: plaquette-path correlator ---
    def plaquette_path_product(self, y_a, x_a, y_b, x_b, route='direct'):
        """
        Product of plaquettes along a strip from A to B.
        This IS gauge-invariant (product of B_p along a path of faces).
        """
        L = self.L
        B = self.plaquette_array()
        prod = 1
        cy, cx = y_a, x_a

        if route == 'direct':
            # Go right, then down (or up)
            while cx != x_b:
                prod *= B[cy, cx]
                cx = (cx + 1) % L
            while cy != y_b:
                prod *= B[cy, cx]
                cy = (cy + 1) % L
        elif route == 'reverse':
            # Go down first, then right
            while cy != y_b:
                prod *= B[cy, cx]
                cy = (cy + 1) % L
            while cx != x_b:
                prod *= B[cy, cx]
                cx = (cx + 1) % L

        return prod


# ===================================================
# SIMULATION
# ===================================================

def run_sim(L=16, beta=2.0, n_therm=2000, n_meas=4000, seed=42, verbose=True):
    """Main simulation loop."""
    tc = Z2GaugeTheory(L, beta, seed)

    det_A = (L // 4, L // 4)
    det_B = (3 * L // 4, 3 * L // 4)
    path_len = L // 4

    if verbose:
        print(f"  L={L}, beta={beta}, therm={n_therm}, meas={n_meas}, seed={seed}")
        print(f"  Edges: {2*L*L}, Plaquettes: {L*L}")
        print(f"  Detectors: A={det_A}, B={det_B}, path_len={path_len}")

    # Thermalize
    t0 = time.time()
    for i in range(n_therm):
        tc.sweep()
    if verbose:
        print(f"  Thermalized in {time.time()-t0:.1f}s")

    # Measure
    data = []
    t0 = time.time()
    for m in range(n_meas):
        tc.sweep()

        Wx, Wy = tc.topological_sector()
        n_any = tc.anyon_count()

        # Gauge-fix and measure Wilson lines
        sh_gf, sv_gf = tc.gauge_fix_tree()
        A = tc.measure_detector(sh_gf, sv_gf, det_A[0], det_A[1], path_len)
        B = tc.measure_detector(sh_gf, sv_gf, det_B[0], det_B[1], path_len)

        # Also: gauge-invariant plaquette at detectors
        Bp = tc.plaquette_array()
        bpA = int(Bp[det_A[0], det_A[1]])
        bpB = int(Bp[det_B[0], det_B[1]])

        data.append({
            'Wx': Wx, 'Wy': Wy, 'n_anyons': n_any,
            'A': A, 'B': B,
            'bpA': bpA, 'bpB': bpB,
        })

        if verbose and (m + 1) % 1000 == 0:
            print(f"    {m+1}/{n_meas}  anyons={n_any:3d}  sector=({Wx:+d},{Wy:+d})")

    if verbose:
        print(f"  Measured in {time.time()-t0:.1f}s")

    return data


# ===================================================
# ANALYSIS
# ===================================================

def correlation_matrix(data):
    """4x4 correlation matrix C[dA, dB] = <A(dA)·B(dB)>."""
    C = np.zeros((4, 4))
    N = len(data)
    for i in range(4):
        for j in range(4):
            C[i, j] = np.mean([r['A'][i] * r['B'][j] for r in data])
    return C


def best_chsh(C):
    """Find max |S| over all direction combos."""
    best = 0
    combo = (0, 1, 0, 1)
    for a1 in range(4):
        for a2 in range(4):
            if a1 == a2: continue
            for b1 in range(4):
                for b2 in range(4):
                    if b1 == b2: continue
                    S = C[a1, b1] - C[a1, b2] + C[a2, b1] + C[a2, b2]
                    if abs(S) > abs(best):
                        best = S
                        combo = (a1, a2, b1, b2)
    return best, combo


def chsh_per_config(data, a1, a2, b1, b2):
    """CHSH value for each individual configuration."""
    return np.array([
        r['A'][a1]*r['B'][b1] - r['A'][a1]*r['B'][b2] +
        r['A'][a2]*r['B'][b1] + r['A'][a2]*r['B'][b2]
        for r in data
    ])


def mutual_info(data, a1, a2, b1, b2, n_shuffle=2000, seed=42):
    """I(topological_sector; S) with permutation test."""
    rng = np.random.default_rng(seed)

    # Sector labels: (Wx, Wy, anyon_pairs_bin)
    sectors = [(r['Wx'], r['Wy'], min(r['n_anyons'] // 2, 10)) for r in data]

    S_vals = chsh_per_config(data, a1, a2, b1, b2)

    # Bin S
    n_bins = 20
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, n_bins + 1)
    S_binned = np.digitize(S_vals, edges)

    def mi(secs, sb):
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

    I_real = mi(sectors, S_binned)

    I_sh = np.array([
        mi([sectors[i] for i in rng.permutation(len(sectors))], S_binned)
        for _ in range(n_shuffle)
    ])

    p = np.mean(I_sh >= I_real) if len(I_sh) > 0 else 1.0
    sigma = (I_real - I_sh.mean()) / I_sh.std() if I_sh.std() > 0 else 0

    # Sector entropy
    N = len(sectors)
    sc = defaultdict(int)
    for s in sectors:
        sc[s] += 1
    H_sec = -sum((c/N) * np.log2(c/N) for c in sc.values())

    sbc = defaultdict(int)
    for b in S_binned:
        sbc[b] += 1
    H_S = -sum((c/N) * np.log2(c/N) for c in sbc.values())

    return {
        'I': I_real, 'p': p, 'sigma': sigma,
        'I_sh_mean': I_sh.mean(), 'I_sh_std': I_sh.std(),
        'H_sec': H_sec, 'H_S': H_S, 'n_sec': len(sc),
    }


def conditional_stats(data, a1, a2, b1, b2):
    """<S> conditioned on (Wx,Wy) and on anyon count."""
    S = chsh_per_config(data, a1, a2, b1, b2)

    # By sector
    by_sec = defaultdict(list)
    for r, s in zip(data, S):
        by_sec[(r['Wx'], r['Wy'])].append(s)

    # By anyon pairs
    by_any = defaultdict(list)
    for r, s in zip(data, S):
        by_any[min(r['n_anyons'] // 2, 15)].append(s)

    return by_sec, by_any


# ===================================================
# MAIN
# ===================================================

def main():
    print("=" * 65)
    print("  Z2 GAUGE THEORY — TORIC CODE SIMULATION v0.5")
    print("  Z2 Lattice Gauge Theory on Torus")
    print("=" * 65)

    L = 16
    beta = 2.0
    n_therm = 2000
    n_meas = 4000
    seed = 42

    # --- Run main simulation ---
    print(f"\n{'-'*65}")
    print("1. MAIN SIMULATION")
    print(f"{'-'*65}")
    data = run_sim(L, beta, n_therm, n_meas, seed)

    # --- Correlation matrix ---
    print(f"\n{'-'*65}")
    print("2. CORRELATION MATRIX  C(dir_A, dir_B)")
    print("   Directions: 0=right, 1=up, 2=left, 3=down")
    print(f"{'-'*65}")
    C = correlation_matrix(data)

    dirs = ['R', 'U', 'L', 'D']
    print(f"\n       ", end='')
    for j in range(4):
        print(f"  B={dirs[j]:>2}", end='')
    print()
    for i in range(4):
        print(f"  A={dirs[i]:>2}", end='')
        for j in range(4):
            print(f"  {C[i,j]:+.4f}", end='')
        print()

    S_best, combo = best_chsh(C)
    a1, a2, b1, b2 = combo
    print(f"\n  Best CHSH:  S = {S_best:+.4f}")
    print(f"  Settings:   a1={dirs[a1]}, a2={dirs[a2]}, b1={dirs[b1]}, b2={dirs[b2]}")
    print(f"  Bell limit: 2.0  |  QM limit: 2sqrt2 ~ 2.828")

    # --- Sector statistics ---
    print(f"\n{'-'*65}")
    print("3. TOPOLOGICAL SECTOR STATISTICS")
    print(f"{'-'*65}")

    sec_counts = defaultdict(int)
    anyon_list = []
    for r in data:
        sec_counts[(r['Wx'], r['Wy'])] += 1
        anyon_list.append(r['n_anyons'])
    anyon_arr = np.array(anyon_list)

    print(f"\n  Sector distribution:")
    for k in sorted(sec_counts.keys()):
        pct = sec_counts[k] / len(data) * 100
        print(f"    ({k[0]:+d}, {k[1]:+d}): {sec_counts[k]:5d}  ({pct:5.1f}%)")

    print(f"\n  Anyons:  mean={anyon_arr.mean():.1f} +/- {anyon_arr.std():.1f}"
          f"   min={anyon_arr.min()}  max={anyon_arr.max()}")

    # Anyon pair histogram
    pair_bins = defaultdict(int)
    for a in anyon_list:
        pair_bins[a // 2] += 1
    print(f"\n  Anyon-pair histogram:")
    for k in sorted(pair_bins.keys())[:12]:
        bar = '#' * (pair_bins[k] * 40 // len(data))
        print(f"    {k:3d} pairs: {pair_bins[k]:5d}  {bar}")

    # --- Conditional statistics ---
    print(f"\n{'-'*65}")
    print("4. CONDITIONAL STATISTICS  <S>(sector) and <S>(anyon_pairs)")
    print(f"{'-'*65}")

    by_sec, by_any = conditional_stats(data, a1, a2, b1, b2)

    print(f"\n  {'Sector':<14} {'N':>6}  {'<S>':>8}  {'+/-SEM':>7}")
    print(f"  {'-'*38}")
    for k in sorted(by_sec.keys()):
        v = by_sec[k]
        n = len(v)
        m = np.mean(v)
        sem = np.std(v) / np.sqrt(n) if n > 1 else 0
        print(f"  ({k[0]:+d},{k[1]:+d})       {n:6d}  {m:+8.4f}  {sem:7.4f}")

    print(f"\n  {'Pairs':>6}  {'N':>6}  {'<S>':>8}  {'+/-SEM':>7}")
    print(f"  {'-'*34}")
    trend_x, trend_y = [], []
    for k in sorted(by_any.keys()):
        v = by_any[k]
        n = len(v)
        m = np.mean(v)
        sem = np.std(v) / np.sqrt(n) if n > 1 else 0
        flag = ''
        if n >= 20:
            trend_x.append(k)
            trend_y.append(m)
        print(f"  {k:6d}  {n:6d}  {m:+8.4f}  {sem:7.4f}  {flag}")

    if len(trend_x) > 2:
        trend_x = np.array(trend_x, dtype=float)
        trend_y = np.array(trend_y)
        r_corr = np.corrcoef(trend_x, trend_y)[0, 1]
        direction = "^pairs -> ^S" if r_corr > 0 else "^pairs -> vS"
        print(f"\n  Trend (N>=20):  r = {r_corr:+.3f}  ({direction})")
        print(f"  {'*** KEY: opposite of XY model if r > 0 ***' if r_corr > 0 else '  (same direction as XY model)'}")

    # --- Mutual Information ---
    print(f"\n{'-'*65}")
    print("5. MUTUAL INFORMATION  I(sector; S)")
    print(f"{'-'*65}")

    mi = mutual_info(data, a1, a2, b1, b2, n_shuffle=2000, seed=seed)

    print(f"\n  I(sector; S) = {mi['I']:.6f} bits")
    print(f"  p-value      = {mi['p']:.4f}")
    print(f"  sigma above null = {mi['sigma']:.1f}sigma")
    print(f"  I_shuffle    = {mi['I_sh_mean']:.6f} +/- {mi['I_sh_std']:.6f}")
    print(f"  H(sector)    = {mi['H_sec']:.3f} bits")
    print(f"  H(S)         = {mi['H_S']:.3f} bits")
    print(f"  N_sectors    = {mi['n_sec']}")

    sig = '***' if mi['sigma'] > 3 else '**' if mi['sigma'] > 2 else '*' if mi['sigma'] > 1.5 else ''
    print(f"\n  Significance: {mi['sigma']:.1f}sigma {sig}")

    # --- Plaquette correlation ---
    print(f"\n{'-'*65}")
    print("6. GAUGE-INVARIANT: PLAQUETTE CORRELATION <B_pA · B_pB>")
    print(f"{'-'*65}")

    bp_corr = np.mean([r['bpA'] * r['bpB'] for r in data])
    bp_A_mean = np.mean([r['bpA'] for r in data])
    bp_B_mean = np.mean([r['bpB'] for r in data])
    bp_connected = bp_corr - bp_A_mean * bp_B_mean
    print(f"\n  <B_pA>       = {bp_A_mean:+.4f}")
    print(f"  <B_pB>       = {bp_B_mean:+.4f}")
    print(f"  <B_pA·B_pB>  = {bp_corr:+.4f}")
    print(f"  Connected:     {bp_connected:+.6f}")

    # --- beta SCAN ---
    print(f"\n{'-'*65}")
    print("7. BETA SCAN — Ordered <-> Disordered transition")
    print(f"{'-'*65}")

    betas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    print(f"\n  {'beta':>5}  {'<S>':>8}  {'max|S|':>8}  {'<any>':>7}  {'I(bits)':>9}  {'sigma':>6}  {'trend_r':>8}")
    print(f"  {'-'*62}")

    for b in betas:
        d = run_sim(L, b, n_therm=1000, n_meas=2000, seed=seed, verbose=False)
        C2 = correlation_matrix(d)
        S2, co2 = best_chsh(C2)
        a1b, a2b, b1b, b2b = co2
        S_arr = chsh_per_config(d, a1b, a2b, b1b, b2b)
        mi2 = mutual_info(d, a1b, a2b, b1b, b2b, n_shuffle=500, seed=seed)
        avg_any = np.mean([r['n_anyons'] for r in d])

        # Trend
        by_any2 = defaultdict(list)
        for r, s in zip(d, S_arr):
            by_any2[min(r['n_anyons'] // 2, 15)].append(s)
        tx, ty = [], []
        for k in sorted(by_any2.keys()):
            if len(by_any2[k]) >= 15:
                tx.append(k)
                ty.append(np.mean(by_any2[k]))
        tr = np.corrcoef(tx, ty)[0, 1] if len(tx) > 2 else float('nan')

        print(f"  {b:5.1f}  {np.mean(S_arr):+8.4f}  {S2:+8.4f}  {avg_any:7.1f}"
              f"  {mi2['I']:9.6f}  {mi2['sigma']:6.1f}  {tr:+8.3f}")

    # --- Comparison with XY v0.1 ---
    print(f"\n{'-'*65}")
    print("8. COMPARISON: Z2 GAUGE vs XY MODEL (v0.1)")
    print(f"{'-'*65}")

    print(f"""
  Property                  XY (v0.1)        Z2 Gauge (v0.5)
  ---------------------------------------------------------
  Model                     Cont. spins      Discrete gauge +/-1
  Topological defects       Vortices          Plaquette anyons
  Sectors                   195 (wind.num.)  4 (Wx,Wy) + anyons
  Best CHSH |S|             0.55             {abs(S_best):.3f}
  I(sector;S)               0.0067 bits      {mi['I']:.4f} bits
  I significance            5.8sigma             {mi['sigma']:.1f}sigma
  Trend ^topo -> ?S          vS (r<0)         {'^S (r>0)' if len(trend_x)>2 and r_corr>0 else 'vS (r<0)' if len(trend_x)>2 else '?'}
  cos(theta) form               r=0.993          N/A (discrete dirs)
  Bell violated?            No               No (classical)
  """)

    print("=" * 65)
    print("  DONE — Z2 gauge theory v0.5 toric code prototype")
    print("=" * 65)


if __name__ == '__main__':
    main()
