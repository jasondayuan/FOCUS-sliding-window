"""GSM8K on DiffusionGemma (transformers) for several decoding modes and batch sizes, one model load.

Usage: python dg_eval.py --data test.parquet --out results/multimodel/dg --runs eb:8:256 eb_window64:8:256 ...
A run is mode:batch_size:num_problems (first N problems); modes are defined in MODES below.
"""
import argparse
import copy
import json
import os
import re
import sys
import time

import pandas as pd
import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

sys.path.insert(0, os.path.dirname(__file__))
import dg_modes  # noqa: E402
from dg_modes import Mode  # noqa: E402

MODES = {
    'eb': Mode(),  # the model's default sampler
    'eb_window64': Mode(window_k=64),
    'eb_focus': Mode(focus=True),
    'eb_focus_window64': Mode(focus=True, window_k=64),
    'eb_stable0.5': Mode(stable_min_conf=0.5),
    'eb_ff': Mode(force_first=True),
    'eb_focus_window64_stable0.5': Mode(focus=True, window_k=64, stable_min_conf=0.5),
    'eb_focus_window64_ff': Mode(focus=True, window_k=64, force_first=True),
    'thr0.9': Mode(rule='thr', threshold=0.9),
    'eb_focus_age2': Mode(focus=True, focus_max_skip=2),
    'eb_focus_window64_age2': Mode(focus=True, window_k=64, focus_max_skip=2),
}
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def extract(text):
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        cand = boxed[-1]
    else:
        nums = re.findall(r'-?[\d,]*\.?\d+', text)
        cand = nums[-1] if nums else ''
    cand = cand.replace(',', '').replace('$', '').replace('\\', '').strip()
    m = re.findall(r'-?\d*\.?\d+', cand)
    return m[-1] if m else cand


def same(pred, gold):
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/diffusiongemma-26B-A4B-it')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    base = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda').eval()
    df = pd.read_parquet(args.data)
    golds = [a.split('####')[-1].strip().replace(',', '') for a in df['answer']]
    prompts = [
        processor.apply_chat_template([{'role': 'user', 'content': q + SUFFIX}], add_generation_prompt=True,
                                      tokenize=True) for q in df['question']
    ]
    prompts = [p['input_ids'] if isinstance(p, dict) else p for p in prompts]
    prompts = [p[0] if p and isinstance(p[0], list) else p for p in prompts]
    pad = base.generation_config.pad_token_id
    eos = set(base.generation_config.eos_token_id)

    def run_batch(model, batch):
        width = max(len(p) for p in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(batch):
            ids[i, width - len(p):] = torch.tensor(p)
            mask[i, width - len(p):] = 1
        with torch.no_grad():
            out = model.generate(ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=args.max_new_tokens)
        gen = out.sequences[:, width:].tolist()
        texts, ntoks = [], []
        for g in gen:
            cut = len(g)
            for j, t in enumerate(g):
                if t in eos:
                    cut = j + 1
                    break
            g = [t for t in g[:cut] if t != pad]
            ntoks.append(len(g))
            texts.append(tok.decode(g, skip_special_tokens=True))
        return texts, ntoks

    # warmup (kernels, allocator) on the stock model
    run_batch(base, prompts[:2])
    for spec in args.runs:
        name, bs, num = spec.split(':')
        bs, num = int(bs), int(num)
        tag = f'{name}_bs{bs}_n{num}'
        if os.path.exists(os.path.join(args.out, f'{tag}.json')):
            print('skip', tag, flush=True)
            continue
        model = copy.copy(base)
        stats = {}
        dg_modes.install(model, MODES[name], stats)
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        texts, ntoks = [], []
        for i in range(0, num, bs):
            t, n = run_batch(model, prompts[i:min(i + bs, num)])
            texts += t
            ntoks += n
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        preds = [extract(t) for t in texts]
        correct = [same(p, g) for p, g in zip(preds, golds[:num])]
        res = dict(tag=tag, mode=name, mode_cfg=vars(MODES[name]), batch_size=bs, n=num,
                   accuracy=sum(correct) / num, wall_s=wall, out_tokens=sum(ntoks), out_tok_per_s=sum(ntoks) / wall,
                   decoder_steps=stats.get('decoder_steps', 0),
                   # canvas tokens per step that went through layers 0-1 / the deep layers, summed over the batch
                   decoder_tokens_l01=stats.get('decoder_tokens_l01', 0),
                   decoder_tokens_deep=stats.get('decoder_tokens_deep', 0),
                   max_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
        with open(os.path.join(args.out, f'{tag}.json'), 'w') as f:
            json.dump(res, f, indent=1)
        with open(os.path.join(args.out, f'{tag}.samples.jsonl'), 'w') as f:
            for t, p, g, c, n in zip(texts, preds, golds, correct, ntoks):
                f.write(json.dumps(dict(output=t, pred=p, gold=g, correct=c, n_tokens=n)) + '\n')
        print(json.dumps({k: v for k, v in res.items() if k != 'mode_cfg'}), flush=True)


if __name__ == '__main__':
    main()
