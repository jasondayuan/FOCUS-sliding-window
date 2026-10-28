# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in per-step timing of DLLM decoding, to find where a step's time goes.

DLLM_STEP_PROFILE=<out.json> records, for every model forward: host time at entry, host time spent inside the forward
(this includes any host sync inside the model, such as FOCUS's at layer 1), the batch size and the tokens fed to
layers 0-1 and to layers 2+.

DLLM_STEP_PROFILE_KERNELS=<start>:<count> also runs torch.profiler over that window of decode steps and stores the
summed GPU kernel time and the top ops (host and device), so GPU busy time can be compared with the step period.
"""
import atexit
import json
import os
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile


class StepProfile:

    def __init__(self, path: str):
        self.path = path
        self.rows = []
        spec = os.getenv('DLLM_STEP_PROFILE_KERNELS', '')
        self.k_start, self.k_count = (int(x) for x in spec.split(':')) if spec else (-1, 0)
        self.prof = None
        self.prof_t0 = 0.0
        self.kernels = None
        self.n_decode = 0
        atexit.register(self.dump)

    def begin(self, is_decoding: bool) -> float:
        if is_decoding and self.n_decode == self.k_start and self.prof is None:
            torch.cuda.synchronize()
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self.prof.start()
            self.prof_t0 = time.perf_counter()
        return time.perf_counter()

    def end(self, t0: float, is_decoding: bool, n_seqs: int, n_in: int, n_deep: int):
        self.rows.append((t0, time.perf_counter() - t0, int(is_decoding), n_seqs, n_in, n_deep))
        if not is_decoding:
            return
        self.n_decode += 1
        if self.prof is not None and self.n_decode == self.k_start + self.k_count:
            torch.cuda.synchronize()
            wall = time.perf_counter() - self.prof_t0
            self.prof.stop()
            self._summarize(wall)
            self.prof = None

    def _summarize(self, wall: float):
        events = self.prof.events()
        # kernels only: record_function ranges also show up on the device timeline and would double count
        gpu_us = sum(e.self_device_time_total for e in events
                     if e.device_type == DeviceType.CUDA and not e.is_user_annotation)
        ranges = {}
        for e in events:
            if e.device_type == DeviceType.CUDA and e.is_user_annotation:
                ranges[e.name] = ranges.get(e.name, 0.0) + e.device_time_total / 1e3
        avg = self.prof.key_averages()
        top_gpu = sorted(avg, key=lambda e: -e.self_device_time_total)[:30]
        top_cpu = sorted(avg, key=lambda e: -e.self_cpu_time_total)[:30]
        self.kernels = dict(steps=self.k_count,
                            wall_s=wall,
                            gpu_kernel_s=gpu_us / 1e6,
                            gpu_ranges_ms=ranges,
                            top_gpu=[(e.key, e.count, e.self_device_time_total / 1e3) for e in top_gpu],
                            top_cpu=[(e.key, e.count, e.self_cpu_time_total / 1e3) for e in top_cpu])

    def dump(self):
        with open(self.path, 'w') as f:
            json.dump(dict(cols=['t0', 'fwd_s', 'decoding', 'n_seqs', 'n_in', 'n_deep'], rows=self.rows,
                           kernels=self.kernels), f)


_path = os.getenv('DLLM_STEP_PROFILE')
STEP_PROFILE = StepProfile(_path) if _path else None
