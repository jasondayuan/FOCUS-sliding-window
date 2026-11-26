"""Tables for unmask_feat3. Usage: python compare.py <results dir>"""
import json
import os
import sys

D = os.path.join(sys.argv[1], 'unmask_feat3')
Z = [('E_base', 'eager: window + FOCUS (tuned)'), ('W_E', 'eager: window only'),
     ('F_E', 'eager: window + FOCUS (defaults)'), ('O_k4', 'opt: window + FOCUS (tuned)'),
     ('O_k4_r2', 'opt: window + FOCUS (tuned, rerun)'), ('W_O', 'opt: window only'),
     ('W_O_r2', 'opt: window only (rerun)'), ('F_O', 'opt: window + FOCUS (defaults)')]
A = [('E_base', 'threshold s=1.0 (base)'), ('E_s0.9', 'threshold s=0.9'), ('E_s0.8', 'threshold s=0.8'),
     ('E_s0.7', 'threshold s=0.7'), ('E_b0.1', 'budget sum(1-p) <= 0.1'), ('E_b0.2', 'budget 0.2'),
     ('E_b0.3', 'budget 0.3'), ('E_b0.5', 'budget 0.5'), ('E_eb0.1', 'EB entropy bound 0.1'),
     ('E_eb0.3', 'EB 0.3'), ('E_eb1.0', 'EB 1.0')]
B = [('E_base', 'eager k=4'), ('E_k8', 'eager k=8'), ('E_adapt', 'eager adaptive 4-8'), ('O_k4', 'optimized k=4'),
     ('O_k4_r2', 'optimized k=4 (rerun)'), ('O_k6', 'optimized k=6'), ('O_k8', 'optimized k=8'),
     ('O_adapt', 'optimized adaptive 4-8'), ('W_O', 'opt window only k=4'), ('W_O_r2', 'opt window only k=4 (rerun)'),
     ('W_O_k6', 'opt window only k=6'), ('W_O_k8', 'opt window only k=8'),
     ('W_O_adapt', 'opt window only adaptive 4-8')]
C = [('O_k4', 'threshold s=1.0'), ('O_k4_r2', 'threshold s=1.0 (rerun)'), ('O_s0.9', 'threshold s=0.9'),
     ('O_s0.8', 'threshold s=0.8'), ('O_b0.2', 'budget 0.2'), ('O_b0.3', 'budget 0.3')]


def load(tag):
    p = os.path.join(D, f'{tag}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def table(rows, ref_tag):
    ref = load(ref_tag)
    print(f"{'config':30s} {'acc%':>5s} {'tok/s':>6s} {'xref':>5s} {'steps':>7s} {'dsteps':>7s} {'tok/step':>8s} "
          f"{'L01/step':>8s} {'deep/stp':>8s} {'out tok':>7s} {'wall s':>6s}")
    for tag, label in rows:
        r = load(tag)
        if r is None:
            print(f'{label:30s} missing')
            continue
        print(f"{label:30s} {100 * r['accuracy']:5.1f} {r['out_tok_per_s']:6.0f} {r['out_tok_per_s'] / ref['out_tok_per_s']:5.2f} "
              f"{r['decode_steps']:7d} {100 * (r['decode_steps'] / ref['decode_steps'] - 1):+6.1f}% "
              f"{r['tokens_per_step']:8.2f} {r['layer01_tokens_per_step']:8.2f} {r['deep_tokens_per_step']:8.2f} "
              f"{r['out_tokens']:7d} {r['wall_s']:6.0f}")


print('0. does FOCUS help (k=4, s=1.0), relative to eager window + FOCUS (tuned)')
table(Z, 'E_base')
print()
print('A. unmasking rule (eager, k=4), relative to threshold s=1.0')
table(A, 'E_base')
print('\nB. window size, relative to optimized k=4')
table(B, 'O_k4')

print('\nC. unmasking rule on the optimized stack (k=4), relative to s=1.0')
table(C, 'O_k4')
