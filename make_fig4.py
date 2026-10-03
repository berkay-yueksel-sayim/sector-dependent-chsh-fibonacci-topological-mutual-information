#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator for fig4_z2_scaling.png (Paper 1a): finite-size scaling of the
sector-shuffle-corrected mutual information I(T;S) at beta=2.0, two Monte-Carlo
engines (v05 legacy row-major, vec vectorized checkerboard), with the three
pre-registered fit models M2 (power law a L^b), M2c (power + asymptote a L^b + c),
and Mexp (exponential a e^{-bL} + c). Mexp is the BIC-preferred model.

Reproduction only -- every value is read from data/stufe2_analyse.json
(summary_per_cell for the points+SEM, fits.{v05,vec} for the model params, BIC,
and predictions). No value is hand-entered.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stufe2_analyse.json"
OUT  = HERE / "figures" / "fig4_z2_scaling.png"

def model(name, p, L):
    L = np.asarray(L, float)
    if name == "M2":   return p[0] * L**p[1]
    if name == "M2c":  return p[0] * L**p[1] + p[2]
    if name == "Mexp": return p[0] * np.exp(-p[1] * L) + p[2]
    raise ValueError(name)

def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    fits = d["fits"]
    engines = [("v05", "o"), ("vec", "s")]        # per figure caption: v05=circles, vec=squares
    colors = {"M2": "tab:orange", "M2c": "tab:green", "Mexp": "tab:red"}
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.constrained_layout.use': True})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    Lgrid = np.logspace(np.log10(7.5), np.log10(66), 400)
    for ax, (eng, marker) in zip(axes, engines):
        f = fits[eng]
        Lv = np.asarray(f["L_values"], float)
        yv = np.asarray(f["y_values"], float)
        sem = np.asarray(f["sem_values"], float)
        # three fit curves (M2/M2c/Mexp); Mexp emphasized (BIC-preferred)
        for m in ("M2", "M2c", "Mexp"):
            bic = f[m]["bic"]
            lw = 2.6 if m == "Mexp" else 1.7
            ax.plot(Lgrid, model(m, f[m]["params"], Lgrid),
                    color=colors[m], lw=lw, label=f"{m} (BIC={bic:.1f})",
                    zorder=3 if m == "Mexp" else 2)
        ax.errorbar(Lv, yv, yerr=sem, fmt=marker, color="tab:blue", ms=6,
                    capsize=3, elinewidth=1.2, mec="tab:blue", zorder=4,
                    label="Data (signal $-$ control)")
        ax.axhline(0, color="0.5", lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlim(7.5, 66)
        ax.set_xticks([8, 16, 32, 48, 64])
        ax.set_xticklabels(["8", "16", "32", "48", "64"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("$L$")
        ax.set_ylabel(r"$I_{\mathrm{corr}} - I_{\mathrm{scrambled}}$ (bits)")
        ax.set_title(f"Engine: {eng}")
        ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    OUT.parent.mkdir(exist_ok=True)
    # save @300 dpi, then strip metadata via a clean PIL re-save (no tEXt/Software)
    tmp = OUT.with_suffix(".raw.png")
    fig.savefig(tmp, dpi=300)
    plt.close(fig)
    from PIL import Image
    im = Image.open(tmp)
    im.save(OUT, dpi=(300, 300))          # PIL save: no matplotlib Software tEXt chunk
    tmp.unlink()
    print(f"wrote {OUT} ({im.size[0]}x{im.size[1]} px)")

if __name__ == "__main__":
    main()
