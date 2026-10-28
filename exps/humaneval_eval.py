# Copyright (c) 2026 SANDS Lab. All rights reserved.
"""HumanEval pass@1 (greedy) + efficiency evaluation for SDAR with FOCUS / sliding-window
variants on the LMDeploy PyTorch engine.

Generated programs are executed against the HumanEval tests in subprocesses with a timeout.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from transformers import AutoTokenizer

from common import add_engine_args, build_engine, efficiency_summary, generate, make_gen_config

INSTRUCTION = ('Complete the following Python function. Return the complete function, including its signature, '
               'in a single ```python code block.\n\n```python\n{prompt}```')


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)(?:```|$)', text, flags=re.S)
    code = next((b for b in blocks if f'def {entry_point}' in b), blocks[0] if blocks else text)
    if f'def {entry_point}' not in code:
        # body-only answer: continue the given signature + docstring
        return prompt + code
    # keep the prompt's imports (e.g. `from typing import List`) in case the answer drops them
    header = '\n'.join(l for l in prompt.splitlines() if l.startswith(('import ', 'from ')))
    return header + '\n' + code


def run_test(program: str, timeout: float = 10.0) -> bool:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'prog.py')
        with open(path, 'w') as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, '-I', path], cwd=d, timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    add_engine_args(ap)
    ap.add_argument('--passes', type=int, default=1,
                    help='send every problem this many times in one run: steady-state throughput (164 problems '
                    'are fewer than a few batches) and pass@1 averaged over the copies')
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.num > 0:
        df = df.iloc[:args.num]
    df = pd.concat([df] * args.passes, ignore_index=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template([{'role': 'user', 'content': INSTRUCTION.format(prompt=p)}], add_generation_prompt=True)
        for p in df['prompt']
    ]

    engine = build_engine(args)
    results, wall = generate(engine, prompts, make_gen_config(args), args.concurrency)

    texts = [tok.decode(ids, skip_special_tokens=True) for ids, _ in results]
    programs = [
        extract_code(t, r.prompt, r.entry_point) + '\n\n' + r.test + f'\ncheck({r.entry_point})\n'
        for t, r in zip(texts, df.itertuples())
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        passed = list(pool.map(run_test, programs))

    summary = dict(tag=args.tag, args=vars(args), n=len(results), accuracy=sum(passed) / len(results))
    summary.update(efficiency_summary(args, results, wall))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'{args.tag}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, f'{args.tag}.samples.jsonl'), 'w') as f:
        for r, t, p, ok, (ids, _) in zip(df.itertuples(), texts, programs, passed, results):
            f.write(json.dumps(dict(task_id=r.task_id, passed=ok, output=t, program=p, n_tokens=len(ids))) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('offset_hist', 'args')}, indent=1))
    engine.close()


if __name__ == '__main__':
    main()
