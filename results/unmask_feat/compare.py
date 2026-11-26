"""Table for the unmasking-rule runs. Usage: python compare.py <results dir>"""
import json
import os
import sys

base = sys.argv[1]
NEW, OLD = os.path.join(base, 'unmask_feat'), os.path.join(base, 'gsm_a6000')
ROWS = [(OLD, 'fw4_mk4_gate1_s1.0', 'FOCUS + window (yesterday, other machine)'),
        (NEW, 'base_orig', 'FOCUS + window (code before the change)'),
        (NEW, 'base_orig_r2', 'FOCUS + window (code before, rerun)'),
        (NEW, 'base', 'FOCUS + window (s=1.0)'),
        (NEW, 'base_r2', 'FOCUS + window (s=1.0, rerun)'),
        (NEW, 'ff', ' + 1 force first'),
        (NEW, 'st', ' + 2 stable unmask'),
        (NEW, 'ff_st', ' + 1 + 2'),
        (NEW, 'base_s0.9', 'threshold only: s=0.9'),
        (NEW, 'base_s0.8', 'threshold only: s=0.8'),
        (NEW, 'base_s0.7', 'threshold only: s=0.7')]


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
for a, b in [('base', 'base_r2'), ('base_orig', 'base_orig_r2'), ('base', 'base_orig'), ('base_r2', 'base_orig_r2')]:
    (_, sa), (_, sb) = load(NEW, a), load(NEW, b)
    if sa is None or sb is None:
        continue
    same = sum(x['output'] == y['output'] for x, y in zip(sa, sb)) / len(sa)
    flips = sum(x['correct'] != y['correct'] for x, y in zip(sa, sb))
    print(f'  {a:>10s} vs {b:<14s} {100 * same:5.1f}%  {flips:3d}')
