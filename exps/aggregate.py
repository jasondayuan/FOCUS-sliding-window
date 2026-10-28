"""Aggregate repeated runs: mean +- std per config, plus within-repeat throughput ratios.

Each repeat directory holds one JSON per config (tag). Repeats of the same sweep usually land on different nodes,
so throughput is also reported relative to a reference config of the same repeat, which cancels node speed.

usage: python exps/aggregate.py <name> <ref_tag> <dir> [<dir> ...]
"""
import glob
import json
import os
import sys

import numpy as np

NUM_LAYERS = 28


def proc_per_out(r):
    if 'proc_per_out_tok' in r:
        return r['proc_per_out_tok']
    return (2 * r['layer01_tokens'] + (NUM_LAYERS - 2) * r['deep_tokens']) / NUM_LAYERS / r['out_tokens']


def load(d):
    return {r['tag']: r for r in (json.load(open(p)) for p in glob.glob(os.path.join(d, '*.json')))}


def aggregate(dirs, ref):
    reps = [load(d) for d in dirs]
    tags = sorted({t for rep in reps for t in rep})
    rows = []
    for t in tags:
        runs = [(rep[t], rep.get(ref)) for rep in reps if t in rep]
        acc = np.array([100 * r['accuracy'] for r, _ in runs])
        tps = np.array([r['out_tok_per_s'] for r, _ in runs])
        ppo = np.array([proc_per_out(r) for r, _ in runs])
        tok_step = np.array([r['tokens_per_step'] for r, _ in runs])
        rel = np.array([r['out_tok_per_s'] / f['out_tok_per_s'] for r, f in runs if f is not None])
        rows.append(dict(tag=t, n=len(runs), acc=acc, tps=tps, ppo=ppo, tok_step=tok_step, rel=rel))
    return rows


def ms(x, spec):
    if x.size == 0:
        return '–'
    if x.size == 1:
        return format(x[0], spec)
    return f'{format(x.mean(), spec)} ± {format(x.std(ddof=1), spec)}'


def to_markdown(rows, ref):
    out = [f'| config | runs | accuracy % | tok/s | tok/s vs {ref} (same repeat) | tok/step | proc/out tok |',
           '|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['tag']} | {r['n']} | {ms(r['acc'], '.1f')} | {ms(r['tps'], '.0f')} | {ms(r['rel'], '.3f')} "
                   f"| {ms(r['tok_step'], '.2f')} | {ms(r['ppo'], '.2f')} |")
    return '\n'.join(out)


def main():
    name, ref, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    rows = aggregate(dirs, ref)
    md = f'## {name} ({len(dirs)} repeat dirs: {", ".join(os.path.basename(d) for d in dirs)})\n\n' + to_markdown(rows, ref)
    print(md)


if __name__ == '__main__':
    main()
