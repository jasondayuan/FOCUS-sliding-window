# Copyright (c) OpenMMLab. All rights reserved.
"""Confidence of the chosen token for dLLM unmasking, without materialising the softmax.

p(token) = exp(logit[token] - logsumexp(logits)), one pass over the vocabulary per row with an online max / sum.
Rows that are not active (not masked, or not computed this step) are skipped and get 0, so the cost follows the
number of masked rows instead of batch x block length.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _token_prob_kernel(LOGITS, TOKENS, ACTIVE, OUT, vocab, stride_row, BLOCK_V: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(ACTIVE + row) == 0:
        tl.store(OUT + row, 0.0)
        return
    base = LOGITS + row * stride_row
    offs = tl.arange(0, BLOCK_V)
    m = tl.full([BLOCK_V], float('-inf'), tl.float32)
    s = tl.zeros([BLOCK_V], tl.float32)
    for start in range(0, vocab, BLOCK_V):
        idx = start + offs
        x = tl.load(base + idx, mask=idx < vocab, other=float('-inf')).to(tl.float32)
        new_m = tl.maximum(m, x)
        # lanes that have only seen -inf keep s = 0 (avoid inf - inf)
        s = tl.where(new_m == float('-inf'), 0.0, s * tl.exp(m - new_m) + tl.exp(x - new_m))
        m = new_m
    row_max = tl.max(m, axis=0)
    total = tl.sum(s * tl.exp(m - row_max), axis=0)
    tok = tl.load(TOKENS + row)
    x_tok = tl.load(base + tok).to(tl.float32)
    tl.store(OUT + row, tl.exp(x_tok - row_max) / total)


def token_prob(logits: torch.Tensor, tokens: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """softmax(logits)[row, tokens[row]] for active rows, 0 elsewhere (float32)."""
    assert logits.dim() == 2 and logits.stride(1) == 1
    rows, vocab = logits.shape
    out = torch.empty((rows, ), dtype=torch.float32, device=logits.device)
    if rows == 0:
        return out
    _token_prob_kernel[(rows, )](logits, tokens.contiguous(), active.to(torch.int8).contiguous(), out, vocab,
                                 logits.stride(0), BLOCK_V=2048, num_warps=8)
    return out
