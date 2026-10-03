#!/usr/bin/env python3
"""
Figure generator for Paper 1a v2.3, "Sector-Dependent CHSH Violation in
Fibonacci Anyons, and Finite-Size Topological--CHSH Mutual Information in 2D
Lattice Models".  Writes fig1-fig3 as PNG to figures/ for LaTeX inclusion
(fig4 has its own generator, make_fig4.py).

Usage: python make_figures_fig1to3.py

Data provenance per panel (what is read vs. what is embedded in this file):
  fig1(a) crossover curves -- read at runtime from
          data/null_calibration_results_vecrep_v2.json (corrected engine,
          21 seeds: seed means after subtracting the circular-shift null,
          components.all_seeds.*.corrected_mean; mean anyon number from
          aggregate_all_seeds.anyons_mean); no analysis values are hard-coded.
  fig1(b) per-seed box plot -- values embedded here; they are the per-seed
          entries of the paper's Table I, which has no separate result JSON.
  fig2    XY sector correlation -- computed in this script (deterministic
          Monte Carlo, seed 42); no external input.
  fig3    model overview -- values embedded here; they are the rows of the
          paper's Table VII (simulation history).
All PNGs are written at dpi 300 and re-saved through clean_save(), which
strips PNG text chunks so no tool or path metadata is embedded.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Output directory (anchored on this file, so the script runs from any cwd).
from pathlib import Path as _P
FIGDIR = str(_P(__file__).resolve().parent / 'figures')
os.makedirs(FIGDIR, exist_ok=True)

from PIL import Image

def clean_save(fig_path, dpi=300):
    """Re-save the just-written PNG with ALL metadata stripped and dpi set.
    Strips matplotlib 'Software' tEXt chunk, any author/path, and rewrites pHYs
    from the given dpi. Pillow's Image.save() with no pnginfo/metadata writes
    only IHDR/pHYs/IDAT/IEND -- no tEXt chunks."""
    with Image.open(fig_path) as im:
        im.load()
        # New image with no .info carried over (drop any text chunks entirely).
        clean = Image.new(im.mode, im.size)
        clean.putdata(list(im.getdata()))
        clean.save(fig_path, format='PNG', dpi=(dpi, dpi))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.constrained_layout.use': True,
    'axes.linewidth': 0.8,
})

ACC = '#3355bb'
RED = '#cc3344'
GRN = '#22aa66'
GLD = '#cc8800'
GRY = '#888888'

# ============================================================
# FIGURE 1: I vs beta + Seed Variation
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                gridspec_kw={'width_ratios': [1.2, 1]})

# Decomposition arrays from the deposited per-seed JSON of the corrected engine
# (data/null_calibration_results_vecrep_v2.json, 21 seeds, circular-shift null
# subtracted; Table II of the paper) -- no hard-coded analysis values.
import json as _json, os as _os
with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        'data', 'null_calibration_results_vecrep_v2.json'),
          encoding='utf-8') as _fh:
    _dec = _json.load(_fh)['decomposition']
_BK = sorted(_dec, key=float)
betas = [float(b) for b in _BK]
_pb = [_dec[b] for b in _BK]
_SEEDS = list(_pb[0]['per_seed'])
assert len(_SEEDS) == 21 and all(set(c['per_seed']) == set(_SEEDS) for c in _pb)
I_full   = [c['components']['all_seeds']['full']['corrected_mean'] for c in _pb]
I_wilson = [c['components']['all_seeds']['wilson']['corrected_mean'] for c in _pb]
I_anyons = [c['components']['all_seeds']['anyons']['corrected_mean'] for c in _pb]
# significance annotation: number of seeds with circular-shift sigma > 3.0
nsig_full = [sum(1 for s in _SEEDS if c['per_seed'][s]['full']['sigma_circ'] > 3.0) for c in _pb]
anyons_avg = [round(c['aggregate_all_seeds']['anyons_mean'], 1) for c in _pb]
# plotted arrays == JSON, recomputed from the per-seed values (guards against silent drift)
for _k, _arr in (('full', I_full), ('wilson', I_wilson), ('anyons', I_anyons)):
    for _c, _v in zip(_pb, _arr):
        _ps = _c['per_seed']
        _m = sum(_ps[s][_k]['I_raw'] - _ps[s][_k]['E_circ'] for s in _SEEDS) / len(_SEEDS)
        assert abs(_m - _v) < 1e-12, (_k, _m, _v)
        assert (sum(1 for s in _SEEDS if _ps[s][_k]['sigma_circ'] > 3.0)
                == _c['components']['all_seeds'][_k]['n_gt3sigma_circ'])
for _c, _a in zip(_pb, anyons_avg):
    assert _a == round(sum(_c['per_seed'][s]['anyons_mean'] for s in _SEEDS) / len(_SEEDS), 1)

ax1.semilogy(betas, [max(x, 1e-4) for x in I_full],
             'o-', color=ACC, lw=2, ms=7,
             label=r'$I$(full sector; $S$)', zorder=3)
ax1.semilogy(betas, [max(x, 1e-4) for x in I_wilson],
             's--', color=RED, lw=1.5, ms=6,
             label=r'$I$(Wilson only; $S$)', zorder=3)
ax1.semilogy(betas, [max(x, 1e-4) for x in I_anyons],
             '^:', color=GRN, lw=1.5, ms=6,
             label=r'$I$(anyons only; $S$)', zorder=3)

ax1.set_xlabel(r'Inverse temperature $\beta$')
ax1.set_ylabel('Mutual Information $I$ (bits)')
ax1.set_ylim(5e-5, 0.2)   # max seed mean ~0.050, room for the annotations
ax1.set_xlim(0.8, 3.3)   # room for the annotation at beta = 3.0
ax1.legend(fontsize=8, loc='upper left')   # lower left holds the clipped points
ax1.set_title('(a) Components of the sector label', fontsize=10)
ax1.grid(True, alpha=0.2)

ax1t = ax1.twinx()
ax1t.bar(betas, anyons_avg, width=0.12, alpha=0.2, color=GRY, zorder=1)
ax1t.set_ylabel('Mean anyons', color=GRY, fontsize=9)
ax1t.tick_params(axis='y', labelcolor=GRY, labelsize=8)
ax1t.set_ylim(0, 50)   # max mean anyon number 30.5; bars stay below the legend

for b, ns in zip(betas, nsig_full):
    if ns > len(_SEEDS) // 2:   # annotate only points significant in the majority of seeds
        ax1.annotate(f'{ns}/{len(_SEEDS)} >3$\\sigma$', xy=(b, I_full[betas.index(b)]),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=7, ha='center', color=ACC)

# Seed-Variation Boxplot
z2_seeds = [0.0397, 0.0176, 0.0178, 0.0099, 0.0301,
            0.0061, 0.0121, 0.0035, 0.0075, 0.0073]
xy_seeds = [0.00281, 0.00053, 0.00067, 0.00192, 0.00059,
            0.00074, 0.00058, 0.00042, 0.00146, 0.00170]

np.random.seed(42)  # Reproducibility for scatter
bp = ax2.boxplot([xy_seeds, z2_seeds], positions=[1, 2], widths=0.5,
                  patch_artist=True, showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='white',
                                 markeredgecolor='black', markersize=5),
                  medianprops=dict(color='black', linewidth=1.5))
bp['boxes'][0].set_facecolor('#bbccee')
bp['boxes'][1].set_facecolor('#aaddcc')

for i, (data, color) in enumerate([(xy_seeds, ACC), (z2_seeds, GRN)]):
    x = np.random.normal(i + 1, 0.06, len(data))
    ax2.scatter(x, data, alpha=0.5, s=25, color=color, zorder=3,
                edgecolors='white', linewidths=0.5)

ax2.axhline(y=0, color=GRY, linestyle=':', alpha=0.5)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['XY\n($\\beta$=1.0)', '$Z_2$ gauge\n($\\beta$=2.0)'],
                     fontsize=9)
ax2.set_ylabel('$I$(sector; $S$) [bits]')
ax2.set_title('(b) Robustness: 10 seeds', fontsize=10)
ax2.grid(True, alpha=0.2, axis='y')

ax2.annotate('4/10 >2$\\sigma$', xy=(1, max(xy_seeds) * 1.05),
             fontsize=8, ha='center', color=RED)
ax2.annotate('9/10 >2$\\sigma$', xy=(2, max(z2_seeds) * 1.05),
             fontsize=8, ha='center', color=GRN)
ax2.set_ylim(top=max(max(z2_seeds), max(xy_seeds)) * 1.25)

_p1 = os.path.join(FIGDIR, 'fig1_crossover_seeds.png')
plt.savefig(_p1, metadata={'Software': None})
plt.close()
clean_save(_p1, dpi=300)
print("Figure 1 saved")

# ============================================================
# FIGURE 2: Delta-C per sector (XY Model)
# ============================================================
class XYModel:
    """Minimal XY model for Figure 2."""
    def __init__(self, L, beta, seed):
        self.L = L
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.uniform(0, 2 * np.pi, (L, L))

    def sweep(self):
        L = self.L
        for y in range(L):
            for x in range(L):
                t_old = self.theta[y, x]
                t_new = t_old + self.rng.normal(0, 1.0)
                dE = 0
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = self.theta[(y + dy) % L, (x + dx) % L]
                    dE += np.cos(t_old - nb) - np.cos(t_new - nb)
                if dE < 0 or self.rng.random() < np.exp(-self.beta * dE):
                    self.theta[y, x] = t_new

    def winding(self):
        L = self.L
        def w(d): return (d + np.pi) % (2 * np.pi) - np.pi
        wx = sum(w(self.theta[0, (x+1) % L] - self.theta[0, x]) for x in range(L))
        wy = sum(w(self.theta[(y+1) % L, 0] - self.theta[y, 0]) for y in range(L))
        return round(wx / (2 * np.pi)), round(wy / (2 * np.pi))

L = 16
beta = 1.0
n_theta = 48
thetas = np.linspace(0, np.pi, n_theta, endpoint=False)
dA, dB = (L // 4, L // 4), (3 * L // 4, 3 * L // 4)

xy = XYModel(L, beta, 42)  # Seed=42: Original-Discovery
for _ in range(1000):
    xy.sweep()

C_soft = defaultdict(lambda: np.zeros(n_theta))
N_sec = defaultdict(int)
for _ in range(8000):
    xy.sweep()
    nx, ny = xy.winding()
    n_abs = abs(nx) + abs(ny)
    tA = xy.theta[dA]
    tB = xy.theta[dB]
    for i, th in enumerate(thetas):
        C_soft[n_abs][i] += np.cos(0 - tA) * np.cos(th - tB)
        C_soft['all'][i] += np.cos(0 - tA) * np.cos(th - tB)
    N_sec[n_abs] += 1
    N_sec['all'] += 1

for k in C_soft:
    C_soft[k] /= N_sec[k]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

colors_n = {0: ACC, 1: RED, 2: GRN, 3: GLD}
for n in [0, 1, 2]:
    if N_sec[n] >= 50:
        ax1.plot(thetas * 180 / np.pi, C_soft[n],
                 label=f'$|n|$={n} ($N$={N_sec[n]})',
                 color=colors_n.get(n, GRY), lw=1.5)
ax1.plot(thetas * 180 / np.pi, -np.cos(thetas) * 0.22,
         '--', color=GRY, lw=1, alpha=0.5, label=r'$-\alpha\cos\theta$')
ax1.set_xlabel(r'$\theta$ (degrees)')
ax1.set_ylabel(r'$C(0, \theta)$')
ax1.set_title(r'(a) Correlation per sector (XY, $\beta$=1.0, soft)', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2)

C_0 = C_soft[0]
for n in [1, 2]:
    if N_sec[n] >= 50:
        dC = C_soft[n] - C_0
        ax2.plot(thetas * 180 / np.pi, dC,
                 label=f'$\\Delta C$: $|n|$={n}$-$$|n|$=0',
                 color=colors_n.get(n, GRY), lw=1.5)
ax2.axhline(0, color=GRY, lw=0.5, ls=':')

for n_res in [2, 3, 4]:
    th_res = 180 / n_res
    ax2.axvline(th_res, color=GLD, alpha=0.3, ls='--', lw=0.8)
    ax2.annotate(f'$\\pi$/{n_res}', xy=(th_res, 0.08),
                 fontsize=7, ha='center', color=GLD)

ax2.set_xlabel(r'$\theta$ (degrees)')
ax2.set_ylabel(r'$\Delta C(\theta) = C_{|n|}(\theta) - C_0(\theta)$')
ax2.set_title('(b) Sector difference (topological signal)', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

_p2 = os.path.join(FIGDIR, 'fig2_sector_correlation.png')
plt.savefig(_p2, metadata={'Software': None})
plt.close()
clean_save(_p2, dpi=300)
print("Figure 2 saved")

# ============================================================
# FIGURE 3: Simulation Overview (inkl. Fibonacci)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 3.2))

versions = [
    'v0.1\nXY', 'v0.2\nXY+coup', 'v0.3\nXY+$\\rho$',
    'v0.4\nXY+trans', 'v0.5\n$Z_2$ cl.',
    'v0.5\n$Z_2$ $\\beta$=3', 'v0.6\nQTC L=2',
    'v0.6\nQTC L=3', 'D1\nIsing', 'D1b\nFibonacci'
]
S_max = [0.55, 0.80, 2.00, 0.44, 0.13, 1.34, 1.41, 0.00, 2.000, 2.811]  # v1.6: 2.554 -> 2.811 (full enumeration + Horodecki)
I_vals = [0.001, None, None, None, 0.015, 0.320, 0.350, 0.000, None, None]
bar_colors = [ACC, ACC, RED, ACC, GRN, GRN, '#8855cc', '#8855cc', GLD, GRN]

bars = ax.bar(range(len(versions)), S_max, color=bar_colors,
              alpha=0.7, edgecolor='white', linewidth=0.5)

# Fibonacci-Balken hervorheben
bars[-1].set_alpha(1.0)
bars[-1].set_edgecolor('black')
bars[-1].set_linewidth(1.5)

# Z2 beta=3 bar (index 5): historical single run, not reproduced by the
# deposited systematic seed set (cf. Table II / crossover validation script).
_IDX_B3 = 5
ax.annotate('single run,\nnot reproduced', xy=(_IDX_B3, S_max[_IDX_B3]),
            xytext=(0, 14), textcoords='offset points',
            fontsize=6.5, ha='center', va='bottom', style='italic',
            color=GRY, linespacing=0.95)

ax.axhline(y=2.0, color=RED, lw=2, ls='--', label='Bell limit $|S|$=2', zorder=5)
ax.axhline(y=2.828, color=RED, lw=1, ls=':', alpha=0.5, label='QM limit $2\\sqrt{2}$')

for i, (v, s) in enumerate(zip(versions, S_max)):
    ax.text(i, s + 0.05, f'{s:.2f}', ha='center', va='bottom',
            fontsize=7.5, fontweight='bold')
    if I_vals[i] is not None:
        ax.text(i, -0.15, f'$I$={I_vals[i]:.3f}', ha='center',
                fontsize=6.5, color=GRY, style='italic')

# Annotation for Fibonacci
ax.annotate('99.4%\nTsirelson', xy=(9, 2.811), xytext=(7.75, 2.90),  # v1.7
            fontsize=7, ha='center', color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1))

# Fine's bound
ax.annotate("Fine's\nbound", xy=(2, 2.0), xytext=(2, 2.35),
            fontsize=8, ha='center', color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1))

ax.set_xticks(range(len(versions)))
ax.set_xticklabels(versions, fontsize=8)
ax.set_ylabel('$|S|_\\mathrm{max}$')
ax.set_title('Simulation History: Maximum CHSH Value per Model', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(-0.3, 3.25)
ax.grid(True, alpha=0.15, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

_p3 = os.path.join(FIGDIR, 'fig3_overview.png')
plt.savefig(_p3, metadata={'Software': None})
plt.close()
clean_save(_p3, dpi=300)
print("Figure 3 saved")

print(f"\nAll figures saved to {FIGDIR}/.")
