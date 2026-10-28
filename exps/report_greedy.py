"""Greedy-decoding results (2026-10-25 afternoon): FOCUS vs FOCUS + window, before and after the engine fixes.

Every point's throughput is normalised to FOCUS slack 1.0 of the same session (report2.summarize).
usage: python exps/report_greedy.py
"""
import glob
import json
import os
import subprocess
import sys

from report2 import HERE, OUT, matched, plot, summarize, table

DEV = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'FOCUS_dev', 'exps')
SETS = [
    # (title, dirs, FOCUS + window prefix, plot file)
    ('greedy, plain engine', 'results_g_rep*', 'fw4_mk4_gate1_s', 'pareto_gsm8k_greedy'),
    ('greedy, engine fixes on (fused unmasking, LM head on masked rows, batched host bookkeeping)',
     'results_gopt_rep*', 'fw_s', 'pareto_gsm8k_greedy_opt'),
    ('greedy, final configuration (all engine fixes + EOS early stop)', 'results_gfinal_rep*', 'fw_s',
     'pareto_gsm8k_greedy_final'),
    ('HumanEval, greedy, engine fixes on (164 problems x 4 passes per run)', 'results_he_gopt_rep*', 'fw_s',
     'pareto_humaneval_greedy_opt'),
]


def series(fw_prefix):
    return [
        ('FOCUS + window k=4 (min-keep 4, eviction gate 1)', fw_prefix, '#2a78d6', 'o'),
        ('FOCUS', 'focus_s', '#eb6834', 's'),
        ('Vanilla (threshold 0.9)', 'vanilla_s', '#eda100', '^'),
    ]


def step_table(dirs):
    rows = []
    for d in dirs:
        out = subprocess.run([sys.executable, os.path.join(DEV, 'prof_report.py'), d], capture_output=True, text=True)
        rows += out.stdout.strip().splitlines()[2 if rows else 0:]
    return '\n'.join(rows)


def breakdown(dirs):
    files = sorted(p for d in dirs for p in glob.glob(os.path.join(d, '*.json')) if not p.endswith('.steps.json'))
    out = subprocess.run([sys.executable, os.path.join(HERE, 'breakdown.py')] + files, capture_output=True, text=True)
    return out.stdout.strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    md = ['# Greedy decoding: FOCUS vs FOCUS + window\n']
    for title, pattern, fw, fname in SETS:
        dirs = sorted(glob.glob(os.path.join(DEV, pattern)))
        dirs = [d for d in dirs if glob.glob(os.path.join(d, '*.json'))]
        if not dirs:
            continue
        stats, n = summarize(dirs)
        if title.startswith('HumanEval'):
            plot(stats, n, 'HumanEval pass@1 (%), mean of 4 passes', fname,
                 'HumanEval (164 x 4 passes per run), greedy, engine fixes on', series=series(fw),
                 labelled=(fw, 'focus_s'), moved={(fw, 0.9): (-8, -6, 'right', 'top'),
                                                  ('focus_s', 1.0): (8, 4, 'left', 'bottom')})
        else:
            plot(stats, n, 'GSM8K accuracy (%)', fname,
                 f'GSM8K (n=1319), {" ".join(title.split()[:4]).rstrip(" (")}', series=series(fw),
                 labelled=(fw, 'focus_s'))
        md += [f'## {title} ({n} session{"s" if n > 1 else ""})\n', table(stats),
               '\n### FOCUS + window at matched accuracy\n', matched(stats, prefix=fw),
               '\n### What each step processes\n', breakdown(dirs),
               '\n### Step time (steady state, full batch)\n', step_table(dirs), '']
    text = '\n'.join(md) + '\n'
    with open(os.path.join(OUT, 'greedy_results.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
