"""Tabulate gsm8k_eval.py results and the decode-offset statistics."""
import glob
import json
import os
import sys

import numpy as np

res_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'results')
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res_dir, '*.json')))]
ref = {r['tag']: r for r in rows}.get('focus')

print(f"{'tag':<20}{'acc':>7}{'wall_s':>8}{'tok/s':>8}{'steps':>8}{'tok/step':>9}"
      f"{'L01/step':>9}{'deep/step':>10}{'L01 vs FOCUS':>13}{'proc/dec':>9}")
for r in rows:
    l01_rel = ''
    # layer-weighted processed tokens per decoded token (2 layers before eviction, the rest after)
    num_layers = 28
    proc_per_dec = (2 * r['layer01_tokens'] + (num_layers - 2) * r['deep_tokens']) / num_layers / r['out_tokens']
    if ref is not None:
        l01_rel = f"{r['layer01_tokens'] / ref['layer01_tokens']:.2f}x"
    print(f"{r['tag']:<20}{r['accuracy'] * 100:>6.2f}%{r['wall_s']:>8.1f}{r['out_tok_per_s']:>8.0f}"
          f"{r['decode_steps']:>8}{r['tokens_per_step']:>9.2f}{r['layer01_tokens_per_step']:>9.2f}"
          f"{r['deep_tokens_per_step']:>10.2f}{l01_rel:>13}{proc_per_dec:>9.2f}")

# hypothesis check: where do newly decoded tokens land relative to the rightmost decoded token?
print('\nOffset of newly decoded token from rightmost previously-decoded position (d = pos - r):')
for r in rows:
    if r['args']['window_k'] != 0:
        continue
    h = np.asarray(r['offset_hist'], dtype=np.float64)
    block = (len(h) - 1) // 2
    total = h.sum()
    if total == 0:
        continue
    d = np.arange(-block, block + 1)
    parts = [f'd<=0 {h[d <= 0].sum() / total:.3f}']
    for k in (1, 2, 4, 8, 16):
        parts.append(f'd>{k} {h[d > k].sum() / total:.3f}')
    print(f"  {r['tag']:<12} n={int(total)}  " + '  '.join(parts))
