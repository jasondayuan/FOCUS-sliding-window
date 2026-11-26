"""Unit checks for force_first / stable_unmask (UnmaskingProcessor and the host-side prediction memory).

Usage: python test_unmask.py   (cuda if available, else cpu)
"""
import types

import numpy as np
import torch

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy
from lmdeploy.pytorch.strategies.dllm.sequence import SchedulerSequenceDLLM
from lmdeploy.pytorch.strategies.dllm.unmasking import UnmaskingProcessor

M, U, C = consts.DLLM_MASKED, consts.DLLM_UNMASKED, consts.DLLM_CACHED
B, V, MASK_ID = 8, 50, 49
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def logits_for(pred, conf):
    """Rows whose argmax is pred with softmax probability conf."""
    n = len(pred)
    lg = torch.zeros(n, V)
    # p = e^a / (e^a + V - 1)  ->  a = log(p (V - 1) / (1 - p))
    a = torch.log(torch.tensor(conf) * (V - 1) / (1 - torch.tensor(conf)))
    lg[torch.arange(n), torch.tensor(pred)] = a
    return lg


# row 0: decoded prefix, pos 5 confident, pos 3 predicts what it predicted last time
# row 1: all masked, nothing confident, top-1 at pos 6, no previous predictions
# row 2: fully decoded block -> retired (cached), nothing to do
# row 3: pos 1 is an unprocessed mask (the caller marks those cached); prev_pred there matches its input id
mask = torch.tensor([[U, U, M, M, M, M, M, M],
                     [M, M, M, M, M, M, M, M],
                     [U, U, U, U, U, U, U, U],
                     [U, C, M, M, M, M, M, M]], dtype=torch.uint8)
pred = [[1, 2, 3, 4, 5, 6, 7, 8],
        [9, 9, 9, 9, 9, 9, 9, 9],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8]]
conf = [[.3, .3, .3, .3, .3, .95, .3, .3],
        [.2, .2, .2, .2, .2, .2, .5, .2],
        [.3] * 8,
        [.3, .3, .3, .3, .6, .3, .3, .3]]
prev = [[-1, -1, 7, 4, 9, 9, 9, 9],
        [-1] * 8,
        [1, 2, 3, 4, 5, 6, 7, 8],
        [-1, MASK_ID, -1, -1, -1, -1, 7, -1]]
input_ids = torch.where(mask == M, MASK_ID, torch.tensor(pred)).flatten()
input_ids[3 * B + 1] = MASK_ID  # the unprocessed mask carries the mask token
logits = torch.cat([logits_for(p, c) for p, c in zip(pred, conf)]).to(dev)
token_ids = logits.argmax(-1)

expect = {
    (False, False): [{5}, {6}, set(), {4}],
    (True, False): [{2, 5}, {0, 6}, set(), {2, 4}],
    (False, True): [{3, 5}, {6}, set(), {4, 6}],
    (True, True): [{2, 3, 5}, {0, 6}, set(), {2, 4, 6}],
}
for (ff, st), want in expect.items():
    cfg = DLLMConfig(block_length=B, unmasking_strategy=UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC, denoising_steps=B,
                     confidence_threshold=0.9, force_first=ff, stable_unmask=st)
    proc = UnmaskingProcessor(cfg)
    out_mask, out_tok, _ = proc(logits, input_ids.to(dev), token_ids.clone(), mask.flatten().clone().to(dev),
                                torch.tensor(prev).flatten().to(dev))
    out_mask = out_mask.view(-1, B).cpu()
    for r in range(4):
        got = {i for i in range(B) if mask[r, i] == M and out_mask[r, i] == U}
        assert got == want[r], (ff, st, r, got, want[r])
        # untouched: everything that was not a mask keeps its state, except the retired row becomes cached
        keep = mask[r] != M
        if r == 2:
            assert (out_mask[r] == C).all()
        else:
            assert (out_mask[r][keep] == mask[r][keep]).all(), (ff, st, r)
    # predictions stay on masked rows, input ids elsewhere
    assert torch.equal(out_tok.cpu(), torch.where(mask.flatten() == M, token_ids.cpu(), input_ids))
    # without prev_pred the stable rule is off
    out2, _, _ = proc(logits, input_ids.to(dev), token_ids.clone(), mask.flatten().clone().to(dev), None)
    want2 = expect[(ff, False)]
    for r in range(4):
        got = {i for i in range(B) if mask[r, i] == M and out2.view(-1, B).cpu()[r, i] == U}
        assert got == want2[r], ('no prev', ff, st, r, got)
print('unmasking: OK')

# host memory: processed masks (argmax) are remembered, unprocessed (mask token) keep the old value, decoded -> -1
seq = types.SimpleNamespace(_prev_pred=np.array([-1, 5, 6, 7, -1, 3], dtype=np.int64), dllm_mask_token=MASK_ID,
                            _stable_ctx=0)
tok = np.array([11, 12, MASK_ID, 14, 15, MASK_ID])
msk = np.array([U, M, M, M, M, U], dtype=np.uint8)
SchedulerSequenceDLLM.record_predictions(seq, tok, msk)
assert seq._prev_pred.tolist() == [-1, 12, 6, 14, 15, -1], seq._prev_pred
print('record_predictions: OK')

# stable_min_conf: every mask repeats its prediction; the floor keeps only the confident ones (top-1 always decodes)
mask1 = torch.tensor([[U, M, M, M, M, M, M, M]], dtype=torch.uint8)
pred1 = [[5, 6, 7, 8, 9, 10, 11, 12]]
conf1 = [[.3, .6, .4, .3, .3, .3, .3, .8]]
lg1 = logits_for(pred1[0], conf1[0]).to(dev)
tok1 = lg1.argmax(-1)
inp1 = torch.where(mask1 == M, MASK_ID, torch.tensor(pred1)).flatten().to(dev)
for floor, want in [(0.0, {1, 2, 3, 4, 5, 6, 7}), (0.5, {1, 7}), (0.7, {7})]:
    cfg = DLLMConfig(block_length=B, unmasking_strategy=UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC, denoising_steps=B,
                     confidence_threshold=0.9, stable_unmask=True, stable_min_conf=floor)
    out, _, _ = UnmaskingProcessor(cfg)(lg1, inp1, tok1.clone(), mask1.flatten().clone().to(dev),
                                        torch.tensor(pred1).flatten().to(dev))
    got = {i for i in range(B) if out.cpu()[i] == U and mask1[0, i] == M}
    assert got == want, (floor, got, want)
print('stable_min_conf: OK')

# stable_ctx (host): a previous prediction only counts once a neighbor within the distance got decoded since then
for dist, want in [(1, [-1, -1, 12, -1, -1, -1]), (2, [-1, -1, 12, 13, -1, -1])]:
    seq = types.SimpleNamespace(_prev_pred=np.full(6, -1, dtype=np.int64), _prev_ctx=np.zeros(6, dtype=np.int16),
                                dllm_mask_token=MASK_ID, _stable_ctx=dist,
                                dllm_mask=np.array([U, M, M, M, M, M], dtype=np.uint8))
    # step: pos 1 decoded, 2..5 predicted and left masked
    SchedulerSequenceDLLM.record_predictions(seq, np.array([1, 11, 12, 13, 14, 15]),
                                             np.array([U, U, M, M, M, M], dtype=np.uint8))
    assert seq._prev_pred.tolist() == [-1, -1, 12, 13, 14, 15]
    seq.dllm_mask = np.array([U, U, M, M, M, M], dtype=np.uint8)  # what update_token_ids leaves
    got = SchedulerSequenceDLLM.stable_prev_pred(seq).tolist()
    assert got == want, (dist, got, want)
    # a step that decodes nothing new near pos 4/5 keeps them ineligible; once pos 3 decodes, pos 4 qualifies (dist 1)
    seq.dllm_mask = np.array([U, U, M, U, M, M], dtype=np.uint8)
    got = SchedulerSequenceDLLM.stable_prev_pred(seq).tolist()
    assert got[4] == 14 and got[5] == (15 if dist == 2 else -1), (dist, got)
print('stable_ctx: OK')

# error budget (1 - conf) and entropy bound (EB-Sampler): rows as below, pos 1 of row 3 is an unprocessed mask
mask3 = torch.tensor([[U, M, M, M, M, M, M, M],
                      [M, M, M, M, M, M, M, M],
                      [U, U, U, U, U, U, U, U],
                      [U, C, M, M, M, M, M, M]], dtype=torch.uint8)
pred3 = [[1, 2, 3, 4, 5, 6, 7, 8]] * 4
conf3 = [[.3, .95, .95, .95, .95, .3, .3, .3],
         [.2, .2, .2, .85, .2, .2, .2, .2],
         [.3] * 8,
         [.3, .99, .99, .99, .5, .99, .3, .3]]
lg3 = torch.cat([logits_for(p, c) for p, c in zip(pred3, conf3)]).to(dev)
tok3 = lg3.argmax(-1)
inp3 = torch.where(mask3 == M, MASK_ID, torch.tensor(pred3)).flatten()
inp3[3 * B + 1] = MASK_ID
inp3 = inp3.to(dev)


def run(**kw):
    cfg = DLLMConfig(block_length=B, unmasking_strategy=UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC, denoising_steps=B,
                     confidence_threshold=0.9, **kw)
    out, _, _ = UnmaskingProcessor(cfg)(lg3, inp3, tok3.clone(), mask3.flatten().clone().to(dev), None)
    out = out.view(-1, B).cpu()
    return [{i for i in range(B) if mask3[r, i] == M and out[r, i] == U} for r in range(4)]


for budget, want in [(0.21, [{1, 2, 3, 4}, {3}, set(), {2, 3, 5}]),
                     (0.12, [{1, 2}, {3}, set(), {2, 3, 5}]),
                     (0.02, [{1}, {3}, set(), {2, 3}]),
                     (0.6, [{1, 2, 3, 4}, {3}, set(), {2, 3, 4, 5}])]:
    got = run(unmask_budget=budget)
    # equal confidences may be taken in either order when the budget cuts between them: compare the chosen confs
    for r in range(4):
        cg = sorted(conf3[r][i] for i in got[r])
        cw = sorted(conf3[r][i] for i in want[r])
        assert cg == cw, (budget, r, got[r], want[r])
print('unmask_budget: OK')

# EB-Sampler reference: ascending entropy, take while sum(H) - max(H) <= bound, at least one
probs = torch.softmax(lg3.float().cpu(), -1)
ent = -(probs * probs.clamp_min(1e-30).log()).sum(-1).view(-1, B)
for bound in [0.05, 0.5, 1.0, 3.0]:
    want = []
    for r in range(4):
        cand = sorted((float(ent[r, i]), i) for i in range(B) if mask3[r, i] == M)
        take, total = set(), 0.0
        for h, i in cand:
            if total > bound:  # sum(H) - max(H) over the taken set = the entropies taken before this one
                break
            total += h
            take.add(i)
        want.append(take)
    got = run(entropy_bound=bound)
    # equal entropies may be taken in either order: compare the chosen entropies, not the positions
    for r in range(4):
        hg = sorted(round(float(ent[r, i]), 4) for i in got[r])
        hw = sorted(round(float(ent[r, i]), 4) for i in want[r])
        assert hg == hw, (bound, r, got[r], want[r])
print('entropy_bound: OK')
