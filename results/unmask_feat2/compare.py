"""Table for the unmasking-rule runs. Usage: python compare.py <results dir>"""
import json
import os
import sys

base = sys.argv[1]
NEW = os.path.join(base, 'unmask_feat2')
ROWS = [(NEW, 'base', 'FOCUS + window (s=1.0)'),
        (NEW, 'base_r2', 'FOCUS + window (s=1.0, rerun)'),
        (NEW, 'st', ' + 2 stable'),
        (NEW, 'st_c0.5', ' + 2 stable, conf >= 0.5'),
        (NEW, 'st_c0.7', ' + 2 stable, conf >= 0.7'),
        (NEW, 'st_n1', ' + 2 stable, neighbor decoded (d=1)'),
        (NEW, 'st_n2', ' + 2 stable, neighbor decoded (d=2)'),
        (NEW, 'st_n1_c0.5', ' + 2 stable, d=1 and conf >= 0.5'),
        (NEW, 'base_s0.9', 'threshold only: s=0.9'),
        (NEW, 'base_s0.8', 'threshold only: s=0.8')]


def load(d, tag):
    p = os.path.join(d, f'{tag}.json')
    if not os.path.exists(p):
        return None, None
    return json.load(open(p)), [json.loads(line) for line in open(os.path.join(d, f'{tag}.samples.jsonl'))]


ref_r, ref_s = load(NEW, 'base')
print(f"{'config':42s} {'acc%':>5s} {'tok/s':>6s} {'xbase':>5s} {'steps':>7s} {'tok/step':>8s} {'proc/out':>8s} "
      f"{'same out':>8s} {'flips':>5s} {'out tok':>7s} {'wall s':>6s}")
for d, tag, label in ROWS:
    r, s = load(d, tag)
    if r is None:
        print(f'{label:42s} missing')
        continue
    same = sum(x['output'] == y['output'] for x, y in zip(s, ref_s)) / len(s)
    flips = sum(x['correct'] != y['correct'] for x, y in zip(s, ref_s))
    print(f"{label:42s} {100 * r['accuracy']:5.1f} {r['out_tok_per_s']:6.0f} {r['out_tok_per_s'] / ref_r['out_tok_per_s']:5.2f} "
          f"{r['decode_steps']:7d} {r['tokens_per_step']:8.2f} {r['proc_per_out_tok']:8.2f} {100 * same:7.1f}% {flips:5d} "
          f"{r['out_tokens']:7d} {r['wall_s']:6.0f}")

print('\npairwise (same outputs / correctness flips):')
for a, b in [('base', 'base_r2')]:
    (_, sa), (_, sb) = load(NEW, a), load(NEW, b)
    if sa is None or sb is None:
        continue
    same = sum(x['output'] == y['output'] for x, y in zip(sa, sb)) / len(sa)
    flips = sum(x['correct'] != y['correct'] for x, y in zip(sa, sb))
    print(f'  {a:>10s} vs {b:<14s} {100 * same:5.1f}%  {flips:3d}')
