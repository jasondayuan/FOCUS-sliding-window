"""Step-time split (median over steady-state full-batch decode steps) for FOCUS and FOCUS + window, slack 1.0,
as the engine fixes are added. Output: exps/report/steptime_greedy.png/pdf"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'FOCUS_dev', 'exps')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#1f1f1e', '#55554f', '#e6e5e0'
FWD, OUTSIDE = '#2a78d6', '#a8b3bf'
STAGES = [('plain engine', 'results_g_rep1', 'focus_s1.0', 'fw4_mk4_gate1_s1.0'),
          ('+ fused unmasking', 'results_g_rep1', 'focus_fu_s1.0', 'fw4_mk4_gate1_fu_s1.0'),
          ('+ LM head on masked rows\n+ batched host bookkeeping', 'results_gopt_rep1', 'focus_s1.0', 'fw_s1.0')]


def split(path):
    r = np.array(json.load(open(path))['rows'], dtype=float)
    j = np.flatnonzero((r[:-1, 2] == 1) & (r[1:, 2] == 1))
    j = j[j >= len(r) // 10]
    j = j[r[j, 3] >= 0.9 * r[j, 3].max()]
    period = r[j + 1, 0] - r[j, 0]
    ok = period < 5 * np.median(period)
    p, f = 1e3 * np.median(period[ok]), 1e3 * np.median(r[j][ok][:, 1])
    return f, p - f


rows = []
for stage, d, focus, fw in STAGES:
    for name, tag in (('FOCUS', focus), ('FOCUS + window', fw)):
        rows.append((f'{name}', stage, *split(os.path.join(DEV, d, f'{tag}.steps.json'))))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
y = []
pos = 0.0
for i, (name, stage, f, o) in enumerate(rows):
    if i % 2 == 0:
        pos += 0.9 if i else 0.5
        # group header above the pair
        ax.text(0, pos - 0.62, stage.replace('\n', ' '), ha='left', va='center', fontsize=8.5, color=INK,
                fontweight='bold')
    y.append(pos)
    ax.barh(pos, f, color=FWD, height=0.7, edgecolor=SURFACE, linewidth=2, label='model forward' if i == 0 else None)
    ax.barh(pos, o, left=f, color=OUTSIDE, height=0.7, edgecolor=SURFACE, linewidth=2,
            label='outside the forward: LM head, sampling + unmasking, host bookkeeping' if i == 0 else None)
    ax.text(f + o + 1, pos, f'{f + o:.1f} ms', va='center', fontsize=8.5, color=INK2)
    pos += 0.8
ax.set_yticks(y)
ax.set_yticklabels([n for n, *_ in rows], fontsize=8.5, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('time per decode step (ms; median over full-batch steps; GSM8K, slack 1.0, greedy)', color=INK2, fontsize=9)
ax.grid(True, axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False, fontsize=8)
ax.set_title('Where a decode step goes, as the engine fixes are added', color=INK, fontsize=10, loc='left', pad=16)
fig.tight_layout()
out = os.path.join(HERE, 'report', 'steptime_greedy')
fig.savefig(out + '.png', facecolor=SURFACE)
fig.savefig(out + '.pdf', facecolor=SURFACE)
for r in rows:
    print(r)
