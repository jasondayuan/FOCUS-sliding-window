"""GSM8K against an SGLang I-DLM server: fixed concurrency, greedy, no thinking. Writes <tag>.json/.samples.jsonl.

Usage: python idlm_client.py --data test.parquet --out DIR --tag T --concurrency C --num N [--stats-path F]
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def extract(text):
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        cand = boxed[-1]
    else:
        nums = re.findall(r'-?[\d,]*\.?\d+', text.split('</think>')[-1])
        cand = nums[-1] if nums else ''
    cand = cand.replace(',', '').replace('$', '').replace('\\', '').strip()
    m = re.findall(r'-?\d*\.?\d+', cand)
    return m[-1] if m else cand


def same(pred, gold):
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def read_stats(path):
    if not path or not os.path.exists(path):
        return {}
    for _ in range(5):
        try:
            return json.load(open(path))
        except (json.JSONDecodeError, OSError):
            time.sleep(0.1)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--port', type=int, default=30000)
    ap.add_argument('--concurrency', type=int, required=True)
    ap.add_argument('--num', type=int, required=True)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--stats-path', default=None)
    args = ap.parse_args()
    df = pd.read_parquet(args.data).iloc[:args.num]
    golds = [a.split('####')[-1].strip().replace(',', '') for a in df['answer']]
    url = f'http://127.0.0.1:{args.port}/v1/chat/completions'

    def one(q):
        r = requests.post(url, json=dict(model='default', messages=[{'role': 'user', 'content': q + SUFFIX}],
                                         max_tokens=args.max_tokens, temperature=0.0,
                                         chat_template_kwargs={'enable_thinking': False}), timeout=1800).json()
        return r['choices'][0]['message']['content'], r['usage']['completion_tokens'], r['choices'][0]['finish_reason']

    one('What is 1 + 1?')  # warmup
    time.sleep(1.0)
    s0 = read_stats(args.stats_path)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        res = list(pool.map(one, df['question']))
    wall = time.perf_counter() - t0
    time.sleep(1.0)
    s1 = read_stats(args.stats_path)
    preds = [extract(c) for c, _, _ in res]
    correct = [same(p, g) for p, g in zip(preds, golds)]
    out_tok = sum(n for _, n, _ in res)
    d = {k: s1.get(k, 0) - s0.get(k, 0) for k in ('total_forwards', 'total_tokens', 'decode_reqs', 'accept_count',
                                                 'reject_count')}
    summary = dict(tag=args.tag, concurrency=args.concurrency, n=args.num, accuracy=sum(correct) / args.num,
                   wall_s=wall, out_tokens=out_tok, out_tok_per_s=out_tok / wall,
                   hit_max_tokens=sum(f == 'length' for _, _, f in res), stats_delta=d,
                   tokens_per_req_step=d['total_tokens'] / d['decode_reqs'] if d.get('decode_reqs') else None)
    os.makedirs(args.out, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out, f'{args.tag}.json'), 'w'), indent=1)
    with open(os.path.join(args.out, f'{args.tag}.samples.jsonl'), 'w') as f:
        for (c, n, fin), p, g, ok in zip(res, preds, golds, correct):
            f.write(json.dumps(dict(output=c, pred=p, gold=g, correct=ok, n_tokens=n, finish=fin)) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
