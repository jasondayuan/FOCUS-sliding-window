# Copyright (c) 2026 SANDS Lab. All rights reserved.
"""GSM8K accuracy + efficiency evaluation for SDAR with FOCUS / sliding-window
variants on the LMDeploy PyTorch engine.

Writes one JSON summary (and per-sample generations) per configuration.
"""
import argparse
import json
import os
import re

import pandas as pd
from transformers import AutoTokenizer

from common import add_engine_args, build_engine, efficiency_summary, generate, make_gen_config

PROMPT_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def extract_pred(text: str):
    boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    cand = boxed[-1] if boxed else text
    nums = re.findall(r'-?\d[\d,]*\.?\d*', cand.replace('$', ''))
    if not nums:
        return None
    return nums[-1].replace(',', '').rstrip('.')


def same_number(pred, gold) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    add_engine_args(ap)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.num > 0:
        df = df.iloc[:args.num]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template([{'role': 'user', 'content': q + PROMPT_SUFFIX}], add_generation_prompt=True)
        for q in df['question']
    ]
    golds = [a.split('####')[-1].strip().replace(',', '') for a in df['answer']]

    engine = build_engine(args)
    results, wall = generate(engine, prompts, make_gen_config(args), args.concurrency)

    correct = 0
    samples = []
    for (ids, _), gold, q in zip(results, golds, df['question']):
        text = tok.decode(ids, skip_special_tokens=True)
        pred = extract_pred(text)
        ok = same_number(pred, gold)
        correct += ok
        samples.append(dict(question=q, gold=gold, pred=pred, correct=bool(ok), output=text, n_tokens=len(ids)))

    n = len(results)
    summary = dict(tag=args.tag, args=vars(args), n=n, accuracy=correct / n)
    summary.update(efficiency_summary(args, results, wall))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'{args.tag}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, f'{args.tag}.samples.jsonl'), 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('offset_hist', 'args')}, indent=1))
    engine.close()


if __name__ == '__main__':
    main()
