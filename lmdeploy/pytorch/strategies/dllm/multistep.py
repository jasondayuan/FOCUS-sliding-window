# Copyright (c) 2026 SANDS Lab. All rights reserved.
"""Device-side delayed-cache bookkeeping, so several decode steps can run per host round trip (dllm_gpu_multistep).

Mirrors the host logic step for step (strategies/dllm/sequence.py):
  DelayedCacheState.update_from_mask   'neighbor' freeze rule, applied to the mask the step started from
  DelayedCacheState.reset              a retired block starts over with every position uncached and a warmup step
  DelayedCacheState.get_processing_indices + SchedulerSequenceDLLM.get_processing_indices / _window_end
                                       the next step feeds the uncached positions (the whole block on warmup or when
                                       nothing is uncached), cut at rightmost decoded + window_k
Only cache_delay='neighbor' without prefix granularity / confidence window / top-n / adaptive window is covered.
"""
import torch

from lmdeploy.pytorch import consts

DLLM_MASKED = consts.DLLM_MASKED


def next_processing(mask_in: torch.Tensor, mask_next: torch.Tensor, is_cached: torch.Tensor, uncached: torch.Tensor,
                    warmup: torch.Tensor, window_k: int):
    """Per-sequence delayed-cache update after one decode step, all [batch, block] / [batch] tensors.

    mask_in: the dllm mask this step was fed (before unmasking). mask_next: the mask the next step is fed (after
    unmasking; retired blocks already replaced by a fresh, fully masked one). is_cached: rows whose block retired.
    Returns (proc [batch, block] bool: positions the next step processes, uncached, warmup) for the step after.
    """
    block = mask_in.size(1)
    # freeze: decoded and right neighbor decoded (the last position has no right neighbor)
    non_mask = mask_in != DLLM_MASKED
    right = torch.ones_like(non_mask)
    right[:, :-1] = non_mask[:, 1:]
    uncached = uncached & ~(non_mask & right)
    # retired block -> fresh state
    uncached = uncached | is_cached[:, None]
    warmup = warmup | is_cached
    full = warmup[:, None] | ~uncached.any(dim=1, keepdim=True)
    proc = full | uncached
    if window_k > 0:
        decoded = mask_next != DLLM_MASKED
        pos = torch.arange(block, device=mask_in.device)
        rightmost = torch.where(decoded, pos[None, :], -1).max(dim=1).values
        end = torch.clamp(rightmost + window_k, max=block - 1)
        proc = proc & (pos[None, :] <= end[:, None])
    # the warmup step is the next one; the step after it is regular again
    return proc, uncached, torch.zeros_like(warmup)
