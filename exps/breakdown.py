"""Per-step breakdown of what each config feeds, keeps and decodes (from the proc_breakdown counters).

usage: python exps/breakdown.py <result.json> [...]
"""
import json
import sys

import numpy as np

from aggregate import proc_per_out

print('| run | acc % | tok/s | out tok/step | masks fed (L0-1) | masks kept (L2+) | decoded / kept | refresh '
      '| proc/out | decodes past r+4 % |')
print('|---|---|---|---|---|---|---|---|---|---|')
for p in sys.argv[1:]:
    r = json.load(open(p))
    if 'proc_breakdown' not in r:
        continue
    s = r['decode_steps']
    dec, still, refresh = r['proc_breakdown']
    kept = r['deep_tokens'] - refresh
    h = np.array(r.get('offset_hist', []))
    far = ''
    if h.size:
        B = (h.size - 1) // 2
        far = f'{100 * h[B + 5:].sum() / h.sum():.1f}'
    name = p.split('exps/')[-1].replace('.json', '')
    print(f"| {name} | {100 * r['accuracy']:.1f} | {r['out_tok_per_s']:.0f} | {r['tokens_per_step']:.2f} | "
          f"{(dec + still) / s:.2f} | {kept / s:.2f} | {dec / max(kept, 1):.2f} | {refresh / s:.2f} | "
          f"{proc_per_out(r):.2f} | {far} |")
