"""Accuracy-throughput plot from repeated runs (mean ± sd over repeats).

usage: python exps/plot_reps.py <out name> <metric label> <results dir> [<results dir> ...]
Throughput is plotted relative to vanilla of the same repeat (removes node-to-node speed differences).
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#1f1f1e', '#55554f', '#8a8a83', '#e6e5e0'
SERIES = [
    # (label, tag prefix, color, marker)
    ('Ours: delayed cache + window k=4', 'dc_w4_s', '#2a78d6', 'o'),
    ('FOCUS', 'focus_s', '#eb6834', 's'),
    ('Ours: delayed cache + window k=2', 'dc_w2_s', '#1baf7a', 'D'),
    ('Vanilla (threshold 0.9)', 'vanilla_s', '#eda100', '^'),
]


def load(d):
    rows = {}
    for p in glob.glob(os.path.join(d, '*.json')):
        r = json.load(open(p))
        rows[r['tag']] = r
    return rows


def main():
    name, ylabel, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    reps = [load(d if os.path.isabs(d) else os.path.join(HERE, d)) for d in dirs]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for label, prefix, color, marker in SERIES:
        tags = sorted({t for rep in reps for t in rep if t.startswith(prefix)}, key=lambda t: -float(t[len(prefix):]))
        pts = []
        for t in tags:
            runs = [rep for rep in reps if t in rep and 'vanilla_s1.0' in rep]
            if not runs:
                continue
            acc = np.array([100 * rep[t]['accuracy'] for rep in runs])
            rel = np.array([rep[t]['out_tok_per_s'] / rep['vanilla_s1.0']['out_tok_per_s'] for rep in runs])
            sd = lambda x: x.std(ddof=1) if len(x) > 1 else 0.0  # noqa: E731
            pts.append((float(t[len(prefix):]), rel.mean(), sd(rel), acc.mean(), sd(acc)))
        if not pts:
            continue
        s, x, xe, y, ye = map(np.array, zip(*pts))
        if len(pts) > 1:
            ax.plot(x, y, color=color, lw=2, zorder=2)
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt='none', ecolor=color, elinewidth=1, alpha=0.6, zorder=2)
        ax.scatter(x, y, s=64, marker=marker, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, label=label)
        if prefix in ('dc_w4_s', 'focus_s'):
            dx, ha = (9, 'left') if prefix == 'dc_w4_s' else (-9, 'right')
            for si, xi, yi in zip(s, x, y):
                ax.annotate(f'{si:g}', (xi, yi), textcoords='offset points', xytext=(dx, 0), ha=ha, va='center',
                            fontsize=8, color=INK2)
    n = len(reps)
    ax.set_xlabel('Throughput relative to vanilla on the same node (×)', color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(f'Accuracy vs. throughput, mean ± sd over {n} repeats\n'
                 'labels = slack (threshold = 0.9 × slack); SDAR-1.7B-Chat-b32, A40, concurrency 128',
                 color=INK, fontsize=10, loc='left')
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    leg = ax.legend(loc='lower left', frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, 'report'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'report', f'{name}.png'), facecolor=SURFACE)
    print('wrote', os.path.join(HERE, 'report', f'{name}.png'))


if __name__ == '__main__':
    main()
