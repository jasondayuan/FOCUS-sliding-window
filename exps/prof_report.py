"""Where a decode step's time goes, from DLLM_STEP_PROFILE dumps (see lmdeploy/pytorch/engine/step_profile.py).

Uses steady-state decode steps only (full batch, after the first 10% of steps). For each config:
- period: time between the starts of consecutive steps (= the real step time),
- in forward: host time inside model_forward, which also waits for the GPU at FOCUS's layer-1 syncs and at the end,
- outside: period - in forward (scheduling, input building, sampling, bookkeeping on the host),
- GPU kernels: summed kernel time per step over the torch.profiler window.

usage: python exps/prof_report.py <dir> [--top N]
"""
import glob
import json
import os
import sys

import numpy as np


def load(p):
    d = json.load(open(p))
    r = np.array(d['rows'], dtype=float)
    return r, d.get('kernels')


def steady(r):
    # period of step j = start of step j+1 - start of step j; only decode steps followed by a decode step
    j = np.flatnonzero((r[:-1, 2] == 1) & (r[1:, 2] == 1))
    j = j[j >= len(r) // 10]
    j = j[r[j, 3] >= 0.9 * r[j, 3].max()]  # (nearly) full batch
    period = r[j + 1, 0] - r[j, 0]
    ok = period < 5 * np.median(period)  # drop pauses (e.g. warmup -> timed run)
    return r[j][ok], period[ok]


def main():
    d = sys.argv[1]
    top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 0
    print('| config | steps | batch | L0-1 tok/step | L2+ tok/step | period ms | in forward ms | outside ms | '
          'GPU kernels ms/step |')
    print('|---|---|---|---|---|---|---|---|---|')
    tops = []
    for p in sorted(glob.glob(os.path.join(d, '*.steps.json'))):
        tag = os.path.basename(p).replace('.steps.json', '')
        r, k = load(p)
        rows, period = steady(r)
        if not len(rows):
            continue
        fwd = rows[:, 1]
        gpu = f"{1e3 * k['gpu_kernel_s'] / k['steps']:.1f}" if k else '–'
        print(f'| {tag} | {len(rows)} | {rows[:, 3].mean():.0f} | {rows[:, 4].mean():.0f} | {rows[:, 5].mean():.0f} | '
              f'{1e3 * np.median(period):.1f} | {1e3 * np.median(fwd):.1f} | {1e3 * np.median(period - fwd):.1f} | '
              f'{gpu} |')
        if k and top:
            tops.append((tag, k))
    for tag, k in tops:
        print(f'\n### {tag}: top GPU ops (ms over {k["steps"]} steps)')
        for name, cnt, ms in k['top_gpu'][:top]:
            print(f'- {ms:8.1f}  x{cnt:<6} {name[:90]}')
        print(f'\n### {tag}: top host ops (self CPU ms over {k["steps"]} steps)')
        for name, cnt, ms in k['top_cpu'][:top]:
            print(f'- {ms:8.1f}  x{cnt:<6} {name[:90]}')


if __name__ == '__main__':
    main()
