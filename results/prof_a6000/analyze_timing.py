"""Step-period stats from a prof_patch timing json (untraced runs).

Usage: python analyze_timing.py <timing.json> <min_bs> <skip> [count]
Uses the decode forwards with batch >= min_bs, skipping the first <skip> of them.
"""
import json
import sys

import numpy as np

r = np.asarray(json.load(open(sys.argv[1]))['rows'])
min_bs, skip = int(sys.argv[2]), int(sys.argv[3])
count = int(sys.argv[4]) if len(sys.argv) > 4 else None
dec = np.flatnonzero((r[:, 1] == 1) & (r[:, 2] >= min_bs))[skip:]
if count:
    dec = dec[:count + 1]
a, b = dec[0], dec[-1]
n = len(dec) - 1
span = r[b, 0] - r[a, 0]
between = r[a:b]
n_pre = int((between[:, 1] == 0).sum())
n_dec_all = int((between[:, 1] == 1).sum())
res = dict(decode_steps=n_dec_all,
           big_decode_steps=n,
           prefill_forwards=n_pre,
           mean_bs=float(between[between[:, 1] == 1, 2].mean()),
           period_ms_per_decode_step=span / n_dec_all * 1e3,
           fwd_decode_host_ms_median=float(np.median(between[between[:, 1] == 1, 3]) * 1e3),
           fwd_prefill_host_ms_median=float(np.median(between[between[:, 1] == 0, 3]) * 1e3) if n_pre else None,
           prefill_ms_per_decode_step=float(between[between[:, 1] == 0, 3].sum() / n_dec_all * 1e3),
           # batch-size independent: sequence-steps (one denoising step of one sequence) per second
           seq_steps_per_s=float(between[between[:, 1] == 1, 2].sum() / span))
print(json.dumps(res, indent=1))
