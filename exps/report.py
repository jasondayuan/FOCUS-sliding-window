"""Build the accuracy-throughput plot and results tables for the report.

Uses the same-node (gpub049) slack sweep in exps/results_slack for the plot,
and exps/results for the k / FOCUS-stacking ablation table.
"""
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'report')
NUM_LAYERS = 28

# reference palette (light), validated: categorical slots 1-4, surface, ink
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
    for p in glob.glob(os.path.join(HERE, d, '*.json')):
        r = json.load(open(p))
        rows[r['tag']] = r
    return rows


def metrics(r):
    a = r['args']
    b = r.get('proc_breakdown') or [0, 0, 0]
    dec = b[0]
    return dict(
        tag=r['tag'],
        threshold=round(a['threshold'] * a.get('slack', 1.0), 3),
        slack=a.get('slack', 1.0),
        acc=100 * r['accuracy'],
        tok_s=r['out_tok_per_s'],
        wall_s=r['wall_s'],
        steps=r['decode_steps'],
        tok_per_step=r['tokens_per_step'],
        l01_per_step=r['layer01_tokens_per_step'],
        deep_per_step=r['deep_tokens_per_step'],
        proc_per_out=(2 * r['layer01_tokens'] + (NUM_LAYERS - 2) * r['deep_tokens']) / NUM_LAYERS / r['out_tokens'],
        idle_per_dec=b[1] / dec if dec else None,
        refresh_per_dec=b[2] / dec if dec else None,
        l01_per_dec=sum(b) / dec if dec else None,
    )


def fmt(v, spec):
    return '–' if v is None else format(v, spec)


def md_table(rows, cols):
    head = '| ' + ' | '.join(c[0] for c in cols) + ' |'
    sep = '|' + '|'.join('---' for _ in cols) + '|'
    body = ['| ' + ' | '.join(fmt(m[c[1]], c[2]) for c in cols) + ' |' for m in rows]
    return '\n'.join([head, sep] + body)


def plot(slack_rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    vanilla = slack_rows.get('vanilla_s1.0')
    if vanilla is not None:
        v = metrics(vanilla)
        ax.axhline(v['acc'], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(1750, v['acc'] - 0.3, 'vanilla accuracy (floor)', color=INK2, fontsize=8.5, va='top')

    for label, prefix, color, marker in SERIES:
        pts = sorted((metrics(r) for t, r in slack_rows.items() if t.startswith(prefix)),
                     key=lambda m: -m['slack'])
        if not pts:
            continue
        xs, ys = [m['tok_s'] for m in pts], [m['acc'] for m in pts]
        if len(pts) > 1:
            ax.plot(xs, ys, color=color, lw=2, zorder=2, solid_capstyle='round')
        ax.scatter(xs, ys, s=90 if len(pts) == 1 else 64, marker=marker, color=color, edgecolors=SURFACE, linewidths=2, zorder=3,
                   label=label)
        if prefix in ('dc_w4_s', 'focus_s'):
            # slack annotations only on the two sweeps being compared
            # ours: labels to the right of each point; FOCUS: to the left, so the two sweeps never collide
            dx, ha = (9, 'left') if prefix == 'dc_w4_s' else (-9, 'right')
            for m in pts:
                ax.annotate(f"{m['slack']:g}", (m['tok_s'], m['acc']), textcoords='offset points',
                            xytext=(dx, 0), ha=ha, va='center', fontsize=8, color=INK2)
        elif prefix == 'vanilla_s':
            ax.annotate('vanilla', (xs[0], ys[0]), textcoords='offset points', xytext=(0, 9), ha='center',
                        va='bottom', fontsize=8.5, color=INK2)

    ax.set_xlim(1350, 3560)
    ax.set_xlabel('Throughput (output tokens / s)', color=INK2)
    ax.set_ylabel('GSM8K accuracy (%)', color=INK2)
    ax.set_title('Accuracy vs. throughput as the confidence threshold is relaxed\n'
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
    fig.savefig(os.path.join(OUT, 'pareto.png'), facecolor=SURFACE)
    fig.savefig(os.path.join(OUT, 'pareto.pdf'), facecolor=SURFACE)


def main():
    os.makedirs(OUT, exist_ok=True)
    slack_rows = load('results_slack')
    ablation_rows = load('results')
    plot(slack_rows)

    order = lambda m: (m['tag'].split('_s')[0], -m['slack'])  # noqa: E731
    slack_m = sorted((metrics(r) for r in slack_rows.values()), key=order)
    abl_m = sorted((metrics(r) for r in ablation_rows.values()), key=lambda m: m['tag'])

    cols = [('config', 'tag', ''), ('threshold', 'threshold', '.2f'), ('acc %', 'acc', '.1f'),
            ('tok/s', 'tok_s', '.0f'), ('decode steps', 'steps', ',d'), ('tok/step', 'tok_per_step', '.2f'),
            ('L0-1 tok/step', 'l01_per_step', '.2f'), ('L2+ tok/step', 'deep_per_step', '.2f'),
            ('proc/out tok', 'proc_per_out', '.2f'), ('L0-1 masked-idle/dec', 'idle_per_dec', '.2f'),
            ('L0-1 refresh/dec', 'refresh_per_dec', '.2f'), ('L0-1 total/dec', 'l01_per_dec', '.2f')]
    abl_cols = cols[:10]
    with open(os.path.join(OUT, 'results.md'), 'w') as f:
        f.write('## Slack sweep (same node gpub049, GSM8K test n=1319, concurrency 128)\n\n')
        f.write(md_table(slack_m, cols) + '\n\n')
        f.write('## Ablations (node gpub057, threshold 0.9, concurrency 128)\n\n')
        f.write(md_table(abl_m, abl_cols[:9]) + '\n')
    with open(os.path.join(OUT, 'results.csv'), 'w') as f:
        keys = list(slack_m[0].keys())
        f.write('sweep,' + ','.join(keys) + '\n')
        for name, ms in (('slack', slack_m), ('ablation', abl_m)):
            for m in ms:
                f.write(name + ',' + ','.join('' if m[k] is None else str(m[k]) for k in keys) + '\n')
    print(open(os.path.join(OUT, 'results.md')).read())


if __name__ == '__main__':
    main()
