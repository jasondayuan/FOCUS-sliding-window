"""Unit checks for the two new SDAR modes: entropy-bounded unmasking and soft force-first.

Usage: python test_new_modes.py   (cuda if available, else cpu)
"""
import torch

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy
from lmdeploy.pytorch.strategies.dllm.unmasking import UnmaskingProcessor

M, U, C = consts.DLLM_MASKED, consts.DLLM_UNMASKED, consts.DLLM_CACHED
B, V, MASK_ID = 8, 64, 63
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


def cfg(**kw):
    return DLLMConfig(block_length=B, unmasking_strategy=UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC, denoising_steps=B,
                      confidence_threshold=0.9, **kw)


def run(proc, logits, mask):
    token_ids = logits.argmax(-1)
    input_ids = torch.where(mask.flatten().cpu() == M, MASK_ID, token_ids.cpu()).to(dev)
    out, _, _ = proc(logits, input_ids, token_ids.clone(), mask.flatten().clone().to(dev), None)
    out = out.view(-1, B).cpu()
    return [{i for i in range(B) if mask[r, i] == M and out[r, i] == U} for r in range(mask.size(0))], out


def eb_reference(logits, mask, bound):
    """Plain-loop version of the rule: ascending entropy, take while sum - max <= bound."""
    ent = torch.distributions.Categorical(logits=logits.float().cpu()).entropy().view(-1, B)
    want = []
    for r in range(mask.size(0)):
        cand = sorted((float(ent[r, i]), i) for i in range(B) if mask[r, i] == M)
        take, total = set(), 0.0
        for e, i in cand:
            total += e
            if total - e > bound:  # e is the running max (ascending order)
                break
            take.add(i)
        want.append(take)
    return want


# random blocks with a mix of sharp and flat rows, masks anywhere, one fully decoded block, one with a cached slot
mask = torch.full((6, B), M, dtype=torch.uint8)
mask[0, :3] = U
mask[2, :] = U
mask[3, 1] = C
mask[4, ::2] = U
scale = torch.tensor([0.5, 2, 4, 8, 16, 32, 64, 128]).repeat(6)[torch.randperm(6 * B)]
logits = (torch.randn(6 * B, V) * scale[:, None]).to(dev)
for bound in (0.05, 0.1, 0.5, 2.0, 1e9):  # 0 exactly would test float32 rounding, not the rule
    got, out = run(UnmaskingProcessor(cfg(entropy_bound=bound)), logits, mask)
    want = eb_reference(logits, mask, bound)
    assert got == want, (bound, got, want)
    for r in range(6):
        has_mask = bool((mask[r] == M).any())
        assert (len(got[r]) >= 1) == has_mask, (bound, r, got[r])  # progress: at least one per block with masks
        keep = mask[r] != M
        if r == 2:
            assert (out[r] == C).all()  # fully decoded block retires as before
        else:
            assert (out[r][keep] == mask[r][keep]).all(), (bound, r)
print('entropy_bound: OK (matches the reference loop for 5 bounds)')


def logits_for(pred, conf):
    n = len(pred)
    lg = torch.zeros(n, V)
    a = torch.log(torch.tensor(conf) * (V - 1) / (1 - torch.tensor(conf)))
    lg[torch.arange(n), torch.tensor(pred)] = a
    return lg.to(dev)


# soft force-first: leftmost mask is pos 2 in both rows, conf .4 (row 0) and .6 (row 1); top-1 elsewhere is pos 5
mask = torch.tensor([[U, U, M, M, M, M, M, M]] * 2, dtype=torch.uint8)
pred = [1, 2, 3, 4, 5, 6, 7, 8]
logits = torch.cat([logits_for(pred, [.3, .3, .4, .3, .3, .8, .3, .3]),
                    logits_for(pred, [.3, .3, .6, .3, .3, .8, .3, .3])])
for kw, want in [(dict(), [{5}, {5}]), (dict(force_first=True), [{2, 5}, {2, 5}]),
                 (dict(force_first=True, force_first_min_conf=0.5), [{5}, {2, 5}])]:
    got, _ = run(UnmaskingProcessor(cfg(**kw)), logits, mask)
    assert got == want, (kw, got, want)
print('force_first_min_conf: OK')
