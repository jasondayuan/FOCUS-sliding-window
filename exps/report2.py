"""Overnight report: repeated slack sweeps (GSM8K, HumanEval) with error bars, plus the new-variant tables.

Throughput differs between nodes by a few percent, so every point's throughput is also normalised to FOCUS
(slack 1.0) of the same repeat before averaging; the plot uses that normalised throughput times the mean
FOCUS throughput over all repeats.
"""
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import proc_per_out  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'report')
DEV = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'FOCUS_dev', 'exps')
REF = 'focus_s1.0'

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#1f1f1e', '#55554f', '#8a8a83', '#e6e5e0'
SERIES = [
    ('Ours: delayed cache + window k=4', 'dc_w4_s', '#2a78d6', 'o'),
    ('FOCUS', 'focus_s', '#eb6834', 's'),
    ('Ours: delayed cache + window k=2', 'dc_w2_s', '#1baf7a', 'D'),
    ('Vanilla (threshold 0.9)', 'vanilla_s', '#eda100', '^'),
]
# FOCUS + window (the combined method) against its two parts
SERIES_FW = [
    ('FOCUS + window k=4', 'fw4_s', '#2a78d6', 'o'),
    ('FOCUS', 'focus_s', '#eb6834', 's'),
    ('Delayed cache + window k=4 (no eviction)', 'dc_w4_s', '#1baf7a', 'D'),
    ('Vanilla (threshold 0.9)', 'vanilla_s', '#eda100', '^'),
]


def load(d):
    return {r['tag']: r for r in (json.load(open(p)) for p in glob.glob(os.path.join(d, '*.json')))
            if 'accuracy' in r}


def summarize(dirs):
    reps = [load(d) for d in dirs if os.path.isdir(d)]
    reps = [r for r in reps if r]
    ref_tps = np.mean([rep[REF]['out_tok_per_s'] for rep in reps if REF in rep])
    out = {}
    for t in sorted({t for rep in reps for t in rep}):
        runs = [rep for rep in reps if t in rep]
        acc = np.array([100 * rep[t]['accuracy'] for rep in runs])
        tps = np.array([rep[t]['out_tok_per_s'] for rep in runs])
        rel = np.array([rep[t]['out_tok_per_s'] / rep[REF]['out_tok_per_s'] for rep in runs if REF in rep])
        ppo = np.array([proc_per_out(rep[t]) for rep in runs])
        out[t] = dict(tag=t, n=len(runs), acc=acc, tps=tps, rel=rel, ppo=ppo, norm_tps=rel * ref_tps,
                      slack=runs[0][t]['args'].get('slack', 1.0),
                      tok_step=np.array([rep[t]['tokens_per_step'] for rep in runs]))
    return out, len(reps)


def sd(x):
    return float(x.std(ddof=1)) if x.size > 1 else 0.0


def ms(x, spec):
    if x.size == 0:
        return '–'
    return format(x.mean(), spec) if x.size == 1 else f'{format(x.mean(), spec)} ± {format(sd(x), spec)}'


def dev_points(dirs, tags, ref_tps):
    """Mean (acc, throughput) of dev-tree runs over several sessions, each normalised to that session's FOCUS."""
    pts = []
    for t in tags:
        acc, tps = [], []
        for d in dirs:
            rows = load(d)
            if t in rows and REF in rows:
                acc.append(100 * rows[t]['accuracy'])
                tps.append(rows[t]['out_tok_per_s'] / rows[REF]['out_tok_per_s'] * ref_tps)
        if acc:
            pts.append((t, np.array(acc), np.array(tps)))
    return pts


def plot(stats, nrep, ylabel, fname, title, extra=None, series=SERIES, labelled=('dc_w4_s', 'focus_s'), moved=None):
    # moved: {(prefix, slack): (dx, dy, ha, va)} to shift individual slack labels clear of nearby points
    moved = moved or {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for label, prefix, color, marker in series:
        pts = sorted((m for t, m in stats.items() if t.startswith(prefix)), key=lambda m: -m['slack'])
        if not pts:
            continue
        xs = [m['norm_tps'].mean() for m in pts]
        ys = [m['acc'].mean() for m in pts]
        ax.errorbar(xs, ys, xerr=[sd(m['norm_tps']) for m in pts], yerr=[sd(m['acc']) for m in pts], fmt='none',
                    ecolor=color, elinewidth=1, capsize=0, alpha=0.6, zorder=2)
        if len(pts) > 1:
            ax.plot(xs, ys, color=color, lw=2, zorder=2)
        ax.scatter(xs, ys, s=64, marker=marker, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, label=label)
        if prefix in labelled:
            # ours above-right, FOCUS below-right: clear of the error bars and of each other
            off, va = ((6, 6), 'bottom') if prefix == labelled[0] else ((6, -6), 'top')
            for m, x, y in zip(pts, xs, ys):
                dx, dy, ha, va_m = moved.get((prefix, m['slack']), (*off, 'left', va))
                ax.annotate(f"{m['slack']:g}", (x, y), textcoords='offset points', xytext=(dx, dy), ha=ha, va=va_m,
                            fontsize=8, color=INK2)
        elif prefix == 'vanilla_s':
            ax.annotate('vanilla', (xs[0], ys[0]), textcoords='offset points', xytext=(0, 9), ha='center',
                        va='bottom', fontsize=8.5, color=INK2)
    if extra:
        label, color, marker, pts = extra
        xs = [p[2].mean() for p in pts]
        ys = [p[1].mean() for p in pts]
        ax.errorbar(xs, ys, xerr=[sd(p[2]) for p in pts], yerr=[sd(p[1]) for p in pts], fmt='none', ecolor=color,
                    elinewidth=1, alpha=0.6, zorder=2)
        ax.scatter(xs, ys, s=64, marker=marker, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, label=label)
        for (t, _, _), x, y in zip(pts, xs, ys):
            ax.annotate(t.split('adapt')[-1].join(['4→', '']), (x, y), textcoords='offset points', xytext=(6, 6),
                        ha='left', va='bottom', fontsize=8, color=INK2)
    ax.set_xlabel('Throughput (output tokens / s, normalised per repeat)', color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    reps = f'mean ± 1 s.d. over {nrep} repeats' if nrep > 1 else 'one run per slack point'
    ax.set_title(f'{title}\n{reps}; labels = slack (threshold = 0.9 × slack)',
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
    fig.savefig(os.path.join(OUT, fname + '.png'), facecolor=SURFACE)
    fig.savefig(os.path.join(OUT, fname + '.pdf'), facecolor=SURFACE)


def curve(stats, prefix):
    """Mean (accuracy, normalised throughput) points of one sweep, ordered by accuracy."""
    pts = sorted(((m['acc'].mean(), m['norm_tps'].mean()) for t, m in stats.items() if t.startswith(prefix)))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def tps_at(stats, prefix, acc):
    """Throughput of a sweep at a given accuracy, linearly interpolated along its mean curve (None if outside)."""
    a, x = curve(stats, prefix)
    if a.size < 2 or not a[0] <= acc <= a[-1]:
        return None
    return float(np.interp(acc, a, x))


def matched(stats, prefix='dc_w4_s'):
    rows = ['| matched accuracy | ours tok/s | FOCUS tok/s | vanilla tok/s | ours ÷ FOCUS | ours ÷ vanilla |',
            '|---|---|---|---|---|---|']
    van = stats.get('vanilla_s1.0')
    targets = [('vanilla default (slack 1.0)', van['acc'].mean() if van else None),
               ('FOCUS default (slack 1.0)', stats[REF]['acc'].mean() if REF in stats else None)]
    for name, acc in targets:
        if acc is None:
            continue
        ours, foc = tps_at(stats, prefix, acc), tps_at(stats, 'focus_s', acc)
        v = van['norm_tps'].mean() if van else None
        f = lambda x: '–' if x is None else f'{x:.0f}'  # noqa: E731
        r = lambda a, b: '–' if a is None or b is None else f'{a / b:.2f}x'  # noqa: E731
        rows.append(f'| {name}: {acc:.1f}% | {f(ours)} | {f(foc)} | {f(v)} | {r(ours, foc)} | {r(ours, v)} |')
    return '\n'.join(rows)


def table(stats, order=None):
    rows = ['| config | runs | accuracy % | tok/s | tok/s ÷ FOCUS (same repeat) | tok/step | proc/out tok |',
            '|---|---|---|---|---|---|---|']
    for t in order or sorted(stats, key=lambda t: (t.split('_s')[0], -stats[t]['slack'])):
        if t not in stats:
            continue
        m = stats[t]
        rows.append(f"| {t} | {m['n']} | {ms(m['acc'], '.1f')} | {ms(m['tps'], '.0f')} | {ms(m['rel'], '.3f')} | "
                    f"{ms(m['tok_step'], '.2f')} | {ms(m['ppo'], '.2f')} |")
    return '\n'.join(rows)


def dev_table(d, ref):
    rows = load(d)
    if not rows:
        return '(no runs)'
    base = rows.get(ref)
    out = ['| config | n | accuracy % | tok/s | tok/s ÷ ref | tok/step | L0-1 tok/dec [dec + idle + refresh] | '
           'proc/out tok |', '|---|---|---|---|---|---|---|---|']
    for t, r in sorted(rows.items()):
        b = r['proc_breakdown']
        dec = max(b[0], 1)
        rel = r['out_tok_per_s'] / base['out_tok_per_s'] if base else float('nan')
        out.append(f"| {t} | {r['n']} | {100 * r['accuracy']:.1f} | {r['out_tok_per_s']:.0f} | {rel:.3f} | "
                   f"{r['tokens_per_step']:.2f} | {sum(b) / dec:.2f} [1 + {b[1] / dec:.2f} + {b[2] / dec:.2f}] | "
                   f"{proc_per_out(r):.2f} |")
    return '\n'.join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    gsm, n_gsm = summarize([os.path.join(HERE, d) for d in ('results_slack', 'results_rep2', 'results_rep3',
                                                             'results_rep4', 'results_rep5')])
    he, n_he = summarize([os.path.join(HERE, f'results_he_rep{i}') for i in range(1, 6)] +
                         [os.path.join(HERE, 'results_he_x4')])
    he_x4, _ = summarize([os.path.join(HERE, 'results_he_x4')])
    plot(gsm, n_gsm, 'GSM8K accuracy (%)', 'pareto_gsm8k_reps', 'GSM8K (n=1319): accuracy vs. throughput')
    if he_x4:
        # throughput only means something at steady state, i.e. in the 4-pass run
        dev_dirs = [os.path.join(DEV, d) for d in ('results_he_dev6', 'results_he_dev8', 'results_he_dev9_rep1',
                                                    'results_he_dev9_rep2')]
        adapt = dev_points(dev_dirs, ['dc_w4_adapt8', 'dc_w4_adapt16', 'dc_w4_adapt32'],
                           he_x4[REF]['tps'].mean())
        plot(he_x4, 1, 'HumanEval pass@1 (%), mean of 4 passes', 'pareto_humaneval_x4',
             'HumanEval (164 problems x 4 passes per run): pass@1 vs. throughput',
             extra=('Ours + adaptive window (slack 1.0; mean ± s.d. of 3-4 runs; label = k range)', '#1baf7a', 'D', adapt))
    md = [f'## GSM8K slack sweep, {n_gsm} repeats\n', table(gsm),
          '\n### Speedup at matched accuracy (GSM8K, interpolated on the mean curves)\n', matched(gsm),
          f'\n## HumanEval slack sweep, {n_he} runs (2 single-pass + 1 four-pass; throughput of single-pass runs '
          'is tail-dominated, use the four-pass table)\n', table(he) if he else '(not run yet)',
          '\n## HumanEval, four passes per run (steady-state throughput)\n', table(he_x4) if he_x4 else '',
          '\n### Speedup at matched accuracy (HumanEval, four-pass run)\n', matched(he_x4) if he_x4 else '',
          '\n## New variants, GSM8K full test (one run each, same session as ref dc_w4_ref)\n',
          dev_table(os.path.join(DEV, 'results_dev'), 'dc_w4_ref'),
          '\n## Smoke tests (first 64 GSM8K problems / 32 HumanEval problems)\n',
          dev_table(os.path.join(DEV, 'results_smoke'), 'dc_w4')]
    text = '\n'.join(md) + '\n'
    with open(os.path.join(OUT, 'overnight_results.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
