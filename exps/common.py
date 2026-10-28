# Copyright (c) 2026 SANDS Lab. All rights reserved.
"""Shared pieces of the eval drivers: engine flags/setup, the async request driver
and aggregation of the per-request dLLM statistics."""
import asyncio
import time

import numpy as np
from transformers import AutoConfig

from lmdeploy.messages import GenerationConfig, PytorchEngineConfig

BLOCK = 32
STOP_IDS = [151645, 151643]


def add_engine_args(ap):
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--num', type=int, default=0, help='first N problems (0 = all)')
    ap.add_argument('--concurrency', type=int, default=128)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('--slack', type=float, default=1.0, help='confidence threshold is scaled to threshold * slack')
    ap.add_argument('--delayed-cache', action='store_true')
    ap.add_argument('--focus', action='store_true')
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--window-k', type=int, default=0)
    ap.add_argument('--prefix-granularity', type=int, default=0)
    ap.add_argument('--cache-delay', default='neighbor', choices=['neighbor', 'one', 'zero', 'conf'],
                    help='when a decoded token stops being re-fed (see dllm_cache_delay)')
    ap.add_argument('--refresh-evict', action='store_true',
                    help='with --focus: re-fed decoded tokens only go through layers 0-1')
    ap.add_argument('--refresh-evict-repeat', action='store_true',
                    help='with --refresh-evict: only the 2nd+ re-feed of a token is cut to layers 0-1')
    ap.add_argument('--window-conf', type=float, default=0.0,
                    help='> 0: place the window from the previous step confidences (see dllm_window_conf)')
    ap.add_argument('--window-k-max', type=int, default=0)
    ap.add_argument('--window-topn', type=int, default=0,
                    help='> 0: feed only the top-n masks of the window by confidence x decoded density')
    ap.add_argument('--window-sigma', type=float, default=2.0)
    ap.add_argument('--window-adapt', action='store_true',
                    help='double / halve the window between window-k and window-k-max from the last step')
    ap.add_argument('--freeze-conf', type=float, default=0.99, help="with --cache-delay conf")
    ap.add_argument('--focus-min-keep', type=int, default=0, help='FOCUS keeps at least this many masks per sequence')
    ap.add_argument('--focus-carry-avg', action='store_true',
                    help="start each block's FOCUS budget from the previous block's average, not 1")
    ap.add_argument('--fast-unmask', action='store_true',
                    help='fused unmasking confidence over masked rows only (no full softmax)')
    ap.add_argument('--masked-logits', action='store_true', help='LM head on processed masked tokens only')
    ap.add_argument('--fast-host', action='store_true', help='batched host bookkeeping for delayed cache / FOCUS')
    ap.add_argument('--eos-early-stop', action='store_true',
                    help='stop once a stop word and its prefix are decoded, not when the whole block is')
    ap.add_argument('--focus-evict-gate', type=float, default=0.0,
                    help='evict only in steps where FOCUS would evict at least this many masks per sequence')
    ap.add_argument('--eager', action='store_true')
    ap.add_argument('--legacy-sampling', action='store_true',
                    help='top-k 50 / temperature 0.8 sampling, as in the runs before greedy decoding was fixed')
    ap.add_argument('--cache-max-entry-count', type=float, default=0.8)


def build_engine(args):
    from lmdeploy.pytorch.engine import Engine
    engine_config = PytorchEngineConfig(
        max_batch_size=args.concurrency,
        eager_mode=args.eager,
        cache_max_entry_count=args.cache_max_entry_count,
        session_len=4096,
        dllm_block_length=BLOCK,
        dllm_denoising_steps=BLOCK,
        dllm_unmasking_strategy='low_confidence_dynamic',
        dllm_confidence_threshold=args.threshold * args.slack,
        dllm_enable_delayed_cache=args.delayed_cache or args.focus,
        dllm_enable_focus=args.focus,
        dllm_focus_alpha=args.alpha,
        dllm_track=True,
        dllm_window_k=args.window_k,
        dllm_window_prefix_granularity=args.prefix_granularity,
        dllm_cache_delay=args.cache_delay,
        dllm_refresh_evict=args.refresh_evict,
        dllm_refresh_evict_repeat=args.refresh_evict_repeat,
        dllm_window_conf=args.window_conf,
        dllm_window_k_max=args.window_k_max,
        dllm_window_topn=args.window_topn,
        dllm_window_sigma=args.window_sigma,
        dllm_window_adapt=args.window_adapt,
        dllm_freeze_conf=args.freeze_conf,
        dllm_focus_min_keep=args.focus_min_keep,
        dllm_focus_carry_avg=args.focus_carry_avg,
        dllm_focus_evict_gate=args.focus_evict_gate,
        dllm_fast_unmask=args.fast_unmask,
        dllm_masked_logits=args.masked_logits,
        dllm_fast_host=args.fast_host,
        dllm_eos_early_stop=args.eos_early_stop,
        max_prefill_token_num=args.concurrency * BLOCK,
    )
    return Engine.from_pretrained(args.model, engine_config=engine_config)


def make_gen_config(args):
    if args.legacy_sampling:
        # what every run before 2026-10-25 16:00 used: the engine ignores do_sample, so GenerationConfig's defaults
        # (top_k=50, temperature=0.8) applied and decoding was sampled
        return GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, stop_token_ids=STOP_IDS)
    # greedy: set what lmdeploy's serving layer (AsyncEngine) sets for do_sample=False; the engine does not do it
    return GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, top_k=1, temperature=1.0,
                            repetition_penalty=1.0, stop_token_ids=STOP_IDS)


async def run_all(engine, prompts, gen_config, concurrency):
    queue = asyncio.Queue()
    for i, ids in enumerate(prompts):
        queue.put_nowait((i, ids))
    results = [None] * len(prompts)

    async def worker(wid):
        inst = engine.create_instance()
        while True:
            try:
                idx, ids = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            session_id = idx + 1
            out_ids, stats = [], None
            gen = inst.async_stream_infer(session_id,
                                          input_ids=ids,
                                          gen_config=gen_config,
                                          sequence_start=True,
                                          sequence_end=True,
                                          stream_output=False)
            try:
                async for outputs in gen:
                    out_ids.extend(outputs.token_ids)
                    m = outputs.req_metrics
                    if m is not None and m.dllm_stats:
                        stats = m.dllm_stats
            finally:
                await gen.aclose()
            await inst.async_end(session_id)
            results[idx] = (list(out_ids), stats)

    await asyncio.gather(*[worker(w) for w in range(concurrency)])
    return results


def generate(engine, prompts, gen_config, concurrency):
    """Untimed warmup on a few prompts, then the timed run; returns (results, wall seconds)."""

    async def warmup_and_run():
        # warmup (kernels / triton autotune); both runs must share one event loop since the engine binds to it
        await run_all(engine, prompts[:min(8, len(prompts))], gen_config, 8)
        t0 = time.perf_counter()
        results = await run_all(engine, prompts, gen_config, concurrency)
        return results, time.perf_counter() - t0

    return asyncio.run(warmup_and_run())


def efficiency_summary(args, results, wall):
    """Throughput and processed-token statistics aggregated over all requests."""
    out_tokens, proc_tokens, input_tokens, steps = 0, 0, 0, 0
    offset_hist = np.zeros(2 * BLOCK + 1, dtype=np.int64)
    proc_breakdown = np.zeros(3, dtype=np.int64)
    for ids, stats in results:
        out_tokens += len(ids)
        if stats:
            proc_tokens += stats.get('processed_tokens', 0)
            input_tokens += stats.get('input_tokens', 0)
            steps += stats.get('decode_steps', 0)
            if 'offset_hist' in stats:
                offset_hist += np.asarray(stats['offset_hist'], dtype=np.int64)
            if 'proc_breakdown' in stats:
                proc_breakdown += np.asarray(stats['proc_breakdown'], dtype=np.int64)
    if out_tokens == 0:
        raise RuntimeError('no tokens generated; engine likely failed (see log above)')
    num_layers = AutoConfig.from_pretrained(args.model, trust_remote_code=True).num_hidden_layers
    return dict(
        wall_s=wall,
        out_tokens=out_tokens,
        out_tok_per_s=out_tokens / wall,
        decode_steps=steps,
        tokens_per_step=out_tokens / max(steps, 1),
        # tokens fed into layers 0-1 (before FOCUS eviction) and into layers 2+ (after eviction)
        layer01_tokens_per_step=input_tokens / max(steps, 1),
        deep_tokens_per_step=proc_tokens / max(steps, 1),
        layer01_tokens=input_tokens,
        deep_tokens=proc_tokens,
        # layer-weighted tokens pushed through the model per output token (layers 0-1 see the
        # pre-eviction set, layers 2+ the post-eviction set)
        proc_per_out_tok=(2 * input_tokens + (num_layers - 2) * proc_tokens) / num_layers / out_tokens,
        offset_hist=offset_hist.tolist(),
        # layer-0/1 tokens per step: [decoded this step, masked and not decoded, already-decoded refresh]
        proc_breakdown=proc_breakdown.tolist(),
    )
