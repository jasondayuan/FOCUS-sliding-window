"""Opt-in NVTX instrumentation of the LMDeploy dLLM step loop (profiling only, no behaviour change).

Env:
  PROF_FULL_BS     decode steps with at least this batch size are counted (default 128)
  PROF_START       full decode step at which cudaProfilerStart is called (-1 = never)
  PROF_COUNT       number of full decode steps to capture
  PROF_TIMING_OUT  json path: host timestamps of every decode forward (for step-period stats)
  PROF_EXIT_AFTER  1 = os._exit once PROF_START + PROF_COUNT full steps were seen (timing runs)
"""
import atexit
import functools
import importlib
import json
import os
import time

import torch
from torch.cuda import nvtx

FULL_BS = int(os.getenv('PROF_FULL_BS', '128'))
START = int(os.getenv('PROF_START', '-1'))
COUNT = int(os.getenv('PROF_COUNT', '0'))
TIMING_OUT = os.getenv('PROF_TIMING_OUT')
EXIT_AFTER = os.getenv('PROF_EXIT_AFTER', '0') == '1'
# cProfile the given host phases (comma separated nvtx names) inside the counted window
CPROFILE_OUT = os.getenv('PROF_CPROFILE')
CPROFILE_NAMES = set(filter(None, os.getenv('PROF_CPROFILE_NAMES', 'update_running,create_inputs').split(',')))
if CPROFILE_OUT:
    import cProfile
    _cprof = cProfile.Profile()
else:
    _cprof = None

_state = dict(n_full=0, active=False)
# (host time at forward entry, is_decoding, batch size, host seconds inside forward)
_rows = []


def _dump():
    if TIMING_OUT:
        with open(TIMING_OUT, 'w') as f:
            json.dump(dict(cols=['t0', 'decoding', 'bs', 'fwd_s'], rows=_rows), f)


atexit.register(_dump)


def _in_window():
    return START <= _state['n_full'] < START + COUNT


def _dump_cprofile():
    if _cprof is None or not CPROFILE_OUT:
        return
    import io
    import pstats
    buf = io.StringIO()
    st = pstats.Stats(_cprof, stream=buf)
    st.sort_stats('tottime').print_stats(30)
    st.sort_stats('cumulative').print_stats(30)
    with open(CPROFILE_OUT, 'w') as f:
        f.write(buf.getvalue())


def _wrap_sync(fn, name):
    profiled = _cprof is not None and name in CPROFILE_NAMES

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        h = nvtx.range_start(name)
        on = profiled and _in_window()
        if on:
            _cprof.enable()
        try:
            return fn(*args, **kwargs)
        finally:
            if on:
                _cprof.disable()
            nvtx.range_end(h)

    return wrapper


def _wrap_async(fn, name):

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        h = nvtx.range_start(name)
        try:
            return await fn(*args, **kwargs)
        finally:
            nvtx.range_end(h)

    return wrapper


def _wrap_forward(fn):

    @functools.wraps(fn)
    def wrapper(model, inputs, *args, **kwargs):
        decoding = bool(inputs.is_decoding)
        bs = int(inputs.seq_length.numel())
        full = decoding and bs >= FULL_BS
        if full and COUNT > 0 and _state['n_full'] == START:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            _state['active'] = True
        t0 = time.perf_counter()
        h = nvtx.range_start('fwd_decode' if decoding else 'fwd_prefill')
        try:
            return fn(model, inputs, *args, **kwargs)
        finally:
            nvtx.range_end(h)
            _rows.append((t0, int(decoding), bs, time.perf_counter() - t0))
            if full:
                _state['n_full'] += 1
                if _state['n_full'] == START + COUNT:
                    if _state['active']:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                        _state['active'] = False
                        # nsys --capture-range-end=stop-shutdown kills the process right after this
                        _dump()
                    if EXIT_AFTER:
                        _dump()
                        _dump_cprofile()
                        os._exit(0)

    return wrapper


# (module, class or None, attribute, nvtx name, async?)
_TARGETS = [
    ('lmdeploy.pytorch.engine.model_agent', 'BaseModelAgent', 'async_sampling_logits', 'sampling', True),
    ('lmdeploy.pytorch.engine.model_agent', 'BaseModelAgent', 'get_output_async', 'wait_out', True),
    ('lmdeploy.pytorch.strategies.dllm.model_agent', 'DLLMModelAgentStrategy', 'post_sampling', 'unmask', False),
    ('lmdeploy.pytorch.strategies.dllm.model_agent', 'DLLMStoppingCriteria', 'step', 'stopping', False),
    ('lmdeploy.pytorch.strategies.dllm.model_agent', 'DLLMModelAgentStrategy', 'make_extra_outputs', 'extra_out',
     False),
    ('lmdeploy.pytorch.engine.engine_loop', 'EngineLoop', '_make_infer_outputs', 'infer_outputs', False),
    ('lmdeploy.pytorch.strategies.dllm.sequence', 'DLLMSequenceStrategy', 'update_running', 'update_running', False),
    ('lmdeploy.pytorch.engine.inputs_maker', 'InputsMakerAsync', '_make_forward_inputs', 'make_inputs', False),
    ('lmdeploy.pytorch.engine.inputs_maker', 'InputsMakerAsync', 'create_model_inputs', 'create_inputs', False),
    ('lmdeploy.pytorch.paging.scheduler', 'Scheduler', 'schedule', 'schedule', False),
]


def install():
    ma = importlib.import_module('lmdeploy.pytorch.engine.model_agent')
    ma.model_forward = _wrap_forward(ma.model_forward)
    for mod_name, cls_name, attr, name, is_async in _TARGETS:
        mod = importlib.import_module(mod_name)
        owner = getattr(mod, cls_name) if cls_name else mod
        fn = getattr(owner, attr)
        setattr(owner, attr, (_wrap_async if is_async else _wrap_sync)(fn, name))


install()
