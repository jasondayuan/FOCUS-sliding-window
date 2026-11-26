"""next_processing (GPU, dllm_gpu_multistep) == the host's delayed-cache + window bookkeeping, on random trajectories.

Host reference: DelayedCacheState (update_from_mask / reset / get_processing_indices) and
SchedulerSequenceDLLM._window_end, exactly as update_running / create_model_inputs use them.
Usage: python test_multistep.py
"""
import types

import numpy as np
import torch

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.strategies.dllm.multistep import next_processing
from lmdeploy.pytorch.strategies.dllm.sequence import DelayedCacheState, SchedulerSequenceDLLM

M, U, C = consts.DLLM_MASKED, consts.DLLM_UNMASKED, consts.DLLM_CACHED
B = 32
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = np.random.default_rng(0)


def host_indices(state, mask, k, fast_host):
    """create_model_inputs -> SchedulerSequenceDLLM.get_processing_indices for one sequence."""
    seq = types.SimpleNamespace(dllm_block_length=B, _window_k=k, dllm_mask=mask, _window_conf=0.0, _last_conf=None,
                                _window_adapt=False, _cur_k=k, _window_k_max=2 * k)
    idx = state.get_processing_indices().numpy()
    end = SchedulerSequenceDLLM._window_end(seq)
    if end < B - 1:
        idx = idx[idx <= end]
    return set(idx.tolist())


checked = 0
for trial in range(60):
    k = int(rng.choice([0, 2, 4, 8]))
    fast_host = bool(trial % 2)
    nseq = 16
    states = [DelayedCacheState.new(B, 0, 'neighbor', fast_host) for _ in range(nseq)]
    masks = [np.full(B, M, dtype=np.uint8) for _ in range(nseq)]
    # round start: the host builds step 1 (consumes the warmup), GPU state starts from there
    proc_host = [host_indices(s, m, k, fast_host) for s, m in zip(states, masks)]
    uncached = torch.stack([s.uncached_positions.clone() for s in states]).to(dev)
    warmup = torch.zeros(nseq, dtype=torch.bool, device=dev)
    for step in range(90):
        mask_in = [m.copy() for m in masks]
        retired = np.zeros(nseq, dtype=bool)
        mask_out = []
        for i in range(nseq):
            m = mask_in[i].copy()
            if (m == U).all():
                retired[i] = True  # unmasking turns a fully decoded block into cached
                m[:] = C
            else:
                cand = [p for p in proc_host[i] if m[p] == M]
                if cand:
                    n = int(rng.integers(1, min(4, len(cand)) + 1))
                    m[rng.choice(cand, size=n, replace=False)] = U
            mask_out.append(m)
        # host: update_running then the next create_model_inputs
        nxt = []
        for i in range(nseq):
            states[i].update_from_mask(mask_in[i])
            if retired[i]:
                states[i].reset()
                masks[i] = np.full(B, M, dtype=np.uint8)
            else:
                masks[i] = mask_out[i]
            nxt.append(host_indices(states[i], masks[i], k, fast_host))
        # GPU
        proc, uncached, warmup = next_processing(
            torch.as_tensor(np.stack(mask_in)).to(dev), torch.as_tensor(np.stack(masks)).to(dev),
            torch.as_tensor(retired).to(dev), uncached, warmup, k)
        for i in range(nseq):
            got = set(torch.nonzero(proc[i]).flatten().tolist())
            assert got == nxt[i], (trial, step, i, k, sorted(got), sorted(nxt[i]))
            assert got, 'empty processing set'
            assert (uncached[i].cpu() == states[i].uncached_positions).all(), (trial, step, i)
        proc_host = nxt
        checked += nseq
print(f'next_processing == host bookkeeping: OK ({checked} sequence-steps, window k in 0/2/4/8, fast_host on/off)')
