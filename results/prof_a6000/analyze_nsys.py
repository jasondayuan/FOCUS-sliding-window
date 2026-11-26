"""Per-decode-step time breakdown from an nsys sqlite export made with prof_patch.py.

Usage: python analyze_nsys.py <report.sqlite> [out.json]

A step is the interval between the starts of two consecutive 'fwd_decode' NVTX ranges. Over all complete
steps in the capture it reports: step period, GPU busy/idle time, what the host was doing while the GPU was
idle, GPU time by the host phase that launched it, and GPU kernel time by category.
"""
import json
import sqlite3
import sys
from collections import defaultdict

import numpy as np

RES_NS = 5_000  # 5 us bins
# host phases, highest priority first (nested / overlapping ranges go to the first match)
PRIORITY = [
    'fwd_decode', 'fwd_prefill', 'sampling', 'unmask', 'stopping', 'extra_out', 'update_running', 'schedule',
    'create_inputs', 'make_inputs', 'infer_outputs', 'wait_out'
]
OTHER = 'other host (loop glue, H2D prep, asyncio)'


def category(name: str) -> str:
    n = name.lower()
    if 'token_prob' in n:
        return 'unmask (fused kernel)'
    if 'focus' in n:
        return 'focus kernels'
    if 'fill_kv' in n or 'fillkv' in n:
        return 'kv cache fill'
    if any(x in n for x in ('attn', 'attention', 'paged', 'flash', '_fwd_', 'ragged', '_reduce_split', 'decode_kernel')):
        return 'attention'
    if any(x in n for x in ('gemm', 'cutlass', 'xmma', 'cublas', 's16816', 'matmul', 'splitk')):
        return 'gemm (linear layers)'
    if 'rms' in n or 'norm' in n:
        return 'norm'
    if 'rotary' in n or 'rope' in n:
        return 'rope'
    if 'silu' in n or 'gelu' in n or 'act' in n:
        return 'activation'
    if any(x in n for x in ('softmax', 'argmax', 'topk', 'sort', 'radix', 'reduce', 'cub::')):
        return 'softmax / reduce / sort'
    return 'elementwise / index / other'


def main():
    db, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    con = sqlite3.connect(db)
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    tables = {r[0] for r in q("select name from sqlite_master where type='table'")}

    nvtx = q('select e.start, e.end, coalesce(e.text, s.value) from NVTX_EVENTS e '
             'left join StringIds s on e.textId = s.id where e.end is not null and e.end > e.start')
    fwd = sorted(s for s, e, t in nvtx if t == 'fwd_decode')
    if len(fwd) < 3:
        raise SystemExit(f'only {len(fwd)} decode forwards captured')
    w0, w1 = fwd[0], fwd[-1]
    n_steps = len(fwd) - 1
    nbins = (w1 - w0) // RES_NS + 1

    def to_mask(intervals):
        acc = np.zeros(nbins + 1, dtype=np.int32)
        for s, e in intervals:
            s, e = max(s, w0), min(e, w1)
            if e <= s:
                continue
            acc[(s - w0) // RES_NS] += 1
            acc[(e - w0) // RES_NS] -= 1
        return np.cumsum(acc)[:nbins] > 0

    # name = short name + demangled name, so templated kernels (e.g. cutlass 'Kernel2') still classify
    kern = q("select k.start, k.end, k.correlationId, s.value || ' ' || d.value from CUPTI_ACTIVITY_KIND_KERNEL k "
             'join StringIds s on k.shortName = s.id join StringIds d on k.demangledName = d.id')
    gpu_iv = [(s, e) for s, e, _, _ in kern]
    mem = []
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        mem = q('select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY')
        gpu_iv += [(s, e) for s, e, _, _ in mem]
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        gpu_iv += q('select start, end from CUPTI_ACTIVITY_KIND_MEMSET')
    busy = to_mask(gpu_iv)

    # host label per bin
    label = np.full(nbins, len(PRIORITY), dtype=np.int16)
    by_name = defaultdict(list)
    for s, e, t in nvtx:
        by_name[t].append((s, e))
    for i in reversed(range(len(PRIORITY))):
        m = to_mask(by_name.get(PRIORITY[i], []))
        label[m] = i
    names = PRIORITY + [OTHER]
    ms = RES_NS / 1e6 / n_steps  # bins -> ms per step

    idle_by = {names[i]: float(((label == i) & ~busy).sum() * ms) for i in range(len(names))}
    host_by = {names[i]: float((label == i).sum() * ms) for i in range(len(names))}

    # GPU time by launching host phase (correlation id -> runtime API call start)
    launch = dict(q('select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME'))
    gpu_by_phase = defaultdict(float)
    gpu_by_cat = defaultdict(float)
    gpu_by_kernel = defaultdict(lambda: [0.0, 0])
    for s, e, cid, name in kern:
        s2, e2 = max(s, w0), min(e, w1)
        if e2 <= s2:
            continue
        dur = (e2 - s2) / 1e6 / n_steps
        t = launch.get(cid)
        ph = names[label[(t - w0) // RES_NS]] if t is not None and w0 <= t < w1 else 'launched outside window'
        gpu_by_phase[ph] += dur
        gpu_by_cat[category(name)] += dur
        gpu_by_kernel[name][0] += dur
        gpu_by_kernel[name][1] += 1
    copy_kind = {1: 'H2D', 2: 'D2H', 8: 'D2D', 10: 'P2P'}
    mem_by = defaultdict(lambda: [0.0, 0, 0])
    for s, e, kind, nbytes in mem:
        if e <= w0 or s >= w1:
            continue
        k = copy_kind.get(kind, str(kind))
        mem_by[k][0] += (min(e, w1) - max(s, w0)) / 1e6 / n_steps
        mem_by[k][1] += 1
        mem_by[k][2] += nbytes

    periods = np.diff(np.asarray(fwd)) / 1e6
    n_prefill = sum(1 for s, e, t in nvtx if t == 'fwd_prefill' and w0 <= s < w1)
    period = (w1 - w0) / 1e6 / n_steps
    res = dict(
        steps=n_steps,
        prefill_forwards_in_window=n_prefill,
        period_ms_mean=period,
        period_ms_median=float(np.median(periods)),
        gpu_busy_ms=float(busy.sum() * ms),
        gpu_idle_ms=float((~busy).sum() * ms),
        gpu_idle_while_host=dict(sorted(idle_by.items(), key=lambda kv: -kv[1])),
        host_time_by_phase=dict(sorted(host_by.items(), key=lambda kv: -kv[1])),
        gpu_ms_by_launch_phase=dict(sorted(gpu_by_phase.items(), key=lambda kv: -kv[1])),
        gpu_ms_by_category=dict(sorted(gpu_by_cat.items(), key=lambda kv: -kv[1])),
        top_kernels=[(k, round(v[0], 3), round(v[1] / n_steps, 1))
                     for k, v in sorted(gpu_by_kernel.items(), key=lambda kv: -kv[1][0])[:15]],
        memcpy_per_step={k: dict(ms=round(v[0], 3), count=round(v[1] / n_steps, 1), kb=round(v[2] / n_steps / 1e3, 1))
                         for k, v in mem_by.items()},
    )
    print(f"steps={n_steps} (+{n_prefill} prefill forwards)  period mean {period:.1f} ms, "
          f"median {res['period_ms_median']:.1f} ms")
    print(f"GPU busy {res['gpu_busy_ms']:.1f} ms/step ({100 * res['gpu_busy_ms'] / period:.0f}%), "
          f"idle {res['gpu_idle_ms']:.1f} ms/step")
    for key in ('gpu_idle_while_host', 'host_time_by_phase', 'gpu_ms_by_launch_phase', 'gpu_ms_by_category'):
        print(f'\n[{key}] ms/step')
        for k, v in res[key].items():
            if v >= 0.05:
                print(f'  {v:7.2f}  {k}')
    print('\n[top kernels] ms/step, launches/step')
    for k, v, c in res['top_kernels']:
        print(f'  {v:7.3f}  {c:6.1f}  {k[:90]}')
    print('\n[memcpy] per step', json.dumps(res['memcpy_per_step']))
    if out:
        with open(out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
