"""Final GSM8K table: accuracy, throughput (relative to FOCUS s=1.0), work per token, and output agreement.

Usage: python compare.py <repo results dir>
"""
import json
import os
import sys

base = sys.argv[1]
OLD, NEW = os.path.join(base, 'gsm_a6000'), os.path.join(base, 'gsm_opt')
ROWS = [
    (OLD, 'vanilla_s1.0', 'baseline (vanilla, 0.9)'),
    (OLD, 'focus_s1.0', 'FOCUS s=1.0'),
    (OLD, 'fw4_mk4_gate1_s1.0', 'FOCUS + window s=1.0'),
    (OLD, 'fw4_mk4_gate1_s0.9', 'FOCUS + window s=0.9'),
    (NEW, 'ref_fw4_s09', 'FOCUS + window s=0.9 (rerun)'),
    (NEW, 'b_flags', ' + PR flags (fast-host/unmask/masked-logits)'),
    (NEW, 'o1_graph', ' + opt1 CUDA graph'),
    (NEW, 'o2_prefill16', ' + opt2 prefill coalescing (16)'),
    (NEW, 'o3_mb64', ' + opt3 micro-batch overlap (64)'),
]


def load(d, tag):
    p = os.path.join(d, f'{tag}.json')
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p))
    s = [json.loads(line) for line in open(os.path.join(d, f'{tag}.samples.jsonl'))]
    return r, s


def agree(a, b):
    same = sum(x['output'] == y['output'] for x, y in zip(a, b))
    flips = sum(x['correct'] != y['correct'] for x, y in zip(a, b))
    return same / len(a), flips


ref_r, ref_s = load(NEW, 'ref_fw4_s09')
focus_r, _ = load(OLD, 'focus_s1.0')
old_ref_r, old_ref_s = load(OLD, 'fw4_mk4_gate1_s0.9')
print(f"{'config':48s} {'acc%':>6s} {'tok/s':>6s} {'xFOCUS':>6s} {'wall s':>6s} {'tok/step':>8s} "
      f"{'proc/out':>8s} {'L01/step':>8s} {'same out':>8s} {'flips':>5s}")
for d, tag, label in ROWS:
    r, s = load(d, tag)
    if r is None:
        print(f'{label:48s} missing')
        continue
    same, flips = agree(s, ref_s) if ref_s else (float('nan'), -1)
    print(f"{label:48s} {100 * r['accuracy']:6.1f} {r['out_tok_per_s']:6.0f} "
          f"{r['out_tok_per_s'] / focus_r['out_tok_per_s']:6.2f} {r['wall_s']:6.0f} {r['tokens_per_step']:8.2f} "
          f"{r['proc_per_out_tok']:8.2f} {r['layer01_tokens_per_step']:8.2f} {100 * same:7.1f}% {flips:5d}")
if old_ref_s and ref_s:
    same, flips = agree(old_ref_s, ref_s)
    print(f'\nA/A noise floor (same config, two runs): {100 * same:.1f}% identical outputs, {flips} correctness flips')
