"""Check the fused token-probability kernel against softmax + gather, and time both on a block-sized batch."""
import time

import torch

from lmdeploy.pytorch.kernels.cuda.dllm_conf import token_prob

torch.manual_seed(0)
for rows, vocab in ((37, 1000), (512, 151936)):
    x = (torch.randn(rows, vocab, device='cuda') * 4).to(torch.bfloat16)
    x[0, :500] = float('-inf')  # banned tokens, as the stop-word filter does
    tok = x.float().argmax(-1)
    active = torch.rand(rows, device='cuda') < 0.5
    ref = torch.where(active, x.softmax(-1).gather(-1, tok[:, None]).flatten().float(), 0.0)
    got = token_prob(x, tok, active)
    print(rows, vocab, 'max abs err', (got - ref).abs().max().item(), 'max ref', ref.max().item())

rows, vocab = int(__import__("os").getenv("TEST_ROWS", "512")), 151936
x = torch.randn(rows, vocab, device='cuda', dtype=torch.bfloat16)
tok = x.argmax(-1)
for frac in (1.0, 0.125):
    active = torch.rand(rows, device='cuda') < frac
    for name, fn in (('softmax+gather', lambda: x.softmax(-1).gather(-1, tok[:, None])),
                     ('fused', lambda: token_prob(x, tok, active))):
        fn(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        print(f'active {frac:.3f} {name}: {(time.perf_counter() - t) / 10 * 1e3:.2f} ms')
