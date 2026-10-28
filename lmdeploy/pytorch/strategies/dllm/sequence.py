# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (HistoryTokenIds, InputEmbeddings, MessageStatus, MultiModalInputs, SamplingParam,
                                       SchedulerSession, UpdateTokenMode, _to_ndarray)
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.config import DLLMConfig

from ..ar.sequence import SchedulerSequenceDefault
from ..base.sequence import SequenceStrategy

SeqList = List['SchedulerSequenceDLLM']

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED
DLLM_MASK_DTYPE = np.uint8


class HistoryDLLMMask(HistoryTokenIds):

    def __init__(self, token_ids: np.ndarray = None, dtype: np.dtype = DLLM_MASK_DTYPE):
        super().__init__(token_ids=token_ids, dtype=dtype)


@dataclass
class FocusInfo:
    mask_local_indices: torch.Tensor
    rightmost_processed: int
    avg_decoded_tokens: float


@dataclass
class DelayedCacheState:
    block_length: int
    uncached_positions: torch.Tensor
    needs_warmup: bool = True
    _full_indices: torch.Tensor = None
    # > 0: only cache the longest decoded prefix, rounded down to this granularity.
    prefix_granularity: int = 0
    # 'neighbor': freeze a decoded token once it has been re-fed and its right neighbor is decoded;
    # 'one': freeze after a single re-feed; 'zero': freeze immediately (keep the mask-stage KV);
    # 'conf': like 'one', but tokens decoded with high confidence are frozen immediately (see freeze_new).
    delay: str = 'neighbor'
    # numpy instead of small torch ops on the per-step path (same results; see --fast-host)
    fast_host: bool = False

    @classmethod
    def new(cls, block_length: int, prefix_granularity: int = 0, delay: str = 'neighbor', fast_host: bool = False):
        # Keep this on regular host memory:
        # - Engine copies indices into its own pinned pool buffers anyway.
        # - Pinning per-sequence buffers adds overhead and can fragment pinned memory.
        uncached = torch.ones((block_length, ), dtype=torch.bool)
        full_indices = torch.arange(block_length, dtype=torch.long)
        return cls(block_length=block_length,
                   uncached_positions=uncached,
                   _full_indices=full_indices,
                   prefix_granularity=prefix_granularity,
                   delay=delay,
                   fast_host=fast_host)

    def reset(self):
        self.uncached_positions.fill_(True)
        self.needs_warmup = True

    def mark_cached(self, ready_mask: torch.Tensor):
        self.uncached_positions &= ~ready_mask

    def update_from_mask(self, dllm_mask: np.ndarray):
        if self.prefix_granularity > 0:
            masked = np.flatnonzero(dllm_mask == DLLM_MASKED)
            prefix_len = int(masked[0]) if masked.size else self.block_length
            prefix_len -= prefix_len % self.prefix_granularity
            ready = torch.zeros((self.block_length, ), dtype=torch.bool)
            ready[:prefix_len] = True
            self.mark_cached(ready)
            return
        if self.fast_host and self.delay == 'neighbor':
            # uncached_positions.numpy() shares memory with the tensor
            non_mask = dllm_mask != DLLM_MASKED
            ready = non_mask.copy()
            ready[:-1] &= non_mask[1:]
            uncached = self.uncached_positions.numpy()
            uncached &= ~ready
            return
        non_mask = torch.from_numpy(dllm_mask != DLLM_MASKED)
        if self.delay != 'neighbor':
            self.mark_cached(non_mask)
            return
        right_neighbor = torch.roll(non_mask, shifts=-1, dims=0)
        right_neighbor[-1] = True
        ready = non_mask & right_neighbor
        self.mark_cached(ready)

    def freeze_new(self, prev_mask: np.ndarray, curr_mask: np.ndarray, conf: np.ndarray, min_conf: float):
        """Freeze tokens decoded this step with confidence >= min_conf, skipping their refresh pass."""
        newly = (prev_mask == DLLM_MASKED) & (curr_mask != DLLM_MASKED) & (conf >= min_conf)
        self.mark_cached(torch.from_numpy(newly))

    def get_processing_indices(self) -> torch.Tensor:
        # Always return a CPU tensor (engine will pack into its pinned pool).
        if self.needs_warmup:
            self.needs_warmup = False
            return self._full_indices

        if self.fast_host:
            indices = torch.from_numpy(np.flatnonzero(self.uncached_positions.numpy()))
        else:
            indices = self.uncached_positions.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            if self.delay in ('zero', 'conf'):
                # block fully decoded and frozen: one token is enough for the step that retires the block
                return self._full_indices[-1:]
            return self._full_indices
        return indices


@dataclass
class FocusState:
    block_length: int
    rightmost_processed: int

    @classmethod
    def new(cls, block_length: int):
        return cls(block_length=block_length, rightmost_processed=-1)

    def reset(self):
        self.rightmost_processed = -1

    def mark_processed(self, processed: int):
        assert processed >= 0 and processed < self.block_length, f'processed={processed}'
        if processed > self.rightmost_processed:
            self.rightmost_processed = processed

    def get_rightmost(self) -> int:
        return self.rightmost_processed


@dataclass
class SchedulerSequenceDLLM(SchedulerSequenceDefault):

    # For dllm
    history_dllm_mask: HistoryDLLMMask = field(default_factory=HistoryDLLMMask)
    dllm_config: DLLMConfig | None = None

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        self._num_valid_ids: int = len(self.history_cache)
        self._strategy: DLLMSequenceStrategy = self._seq_meta.strategy
        self._delayed_cache_state: DelayedCacheState | None = None
        cfg = self.dllm_config
        self._window_k = int(getattr(cfg, 'window_k', 0) or 0) if cfg else 0
        self._window_conf = float(getattr(cfg, 'window_conf', 0.0) or 0.0) if cfg else 0.0
        self._window_k_max = int(getattr(cfg, 'window_k_max', 0) or 0) if cfg else 0
        if self._window_k_max <= 0:
            self._window_k_max = 2 * self._window_k
        self._window_topn = int(getattr(cfg, 'window_topn', 0) or 0) if cfg else 0
        self._window_sigma = float(getattr(cfg, 'window_sigma', 2.0) or 2.0) if cfg else 2.0
        self._window_adapt = bool(getattr(cfg, 'window_adapt', False)) if cfg else False
        # current window size under window_adapt (kept across blocks: the text's regime persists)
        self._cur_k = self._window_k
        # last computed confidence of every block position (the block's first step computes all of them)
        self._last_conf: Optional[np.ndarray] = None
        self._refresh_evict_repeat = bool(getattr(cfg, 'refresh_evict', False)
                                          and getattr(cfg, 'refresh_evict_repeat', False)) if cfg else False
        # how often each decoded position of the block has been re-fed with its real id
        self._refeeds = np.zeros((self.dllm_block_length, ), dtype=np.int16)
        if self._strategy.enable_delayed_cache:
            prefix_granularity = int(getattr(cfg, 'window_prefix_granularity', 0) or 0) if cfg else 0
            delay = getattr(cfg, 'cache_delay', 'neighbor') if cfg else 'neighbor'
            fast_host = bool(getattr(cfg, 'fast_host', False)) if cfg else False
            self._delayed_cache_state = DelayedCacheState.new(self.dllm_block_length, prefix_granularity, delay,
                                                              fast_host)
        focus_enabled = bool(cfg and getattr(cfg, 'enable_focus', False))
        self._focus_enabled = focus_enabled and self.delayed_cache_enabled
        self._focus_state: FocusState | None = FocusState.new(self.dllm_block_length) if self._focus_enabled else None
        self._focus_token_sum: float = 0.0
        self._focus_steps: int = 0
        # FOCUS retain target = max(ceil(alpha x avg), min_keep): raise avg to min_keep / alpha
        alpha = float(getattr(cfg, 'focus_alpha', 1.0) or 1.0) if cfg else 1.0
        self._focus_min_avg = int(getattr(cfg, 'focus_min_keep', 0) or 0) / alpha if cfg else 0.0
        self._focus_carry_avg = bool(getattr(cfg, 'focus_carry_avg', False)) if cfg else False
        self._focus_block_start_avg = 1.0
        self._fast_host = bool(getattr(cfg, 'fast_host', False)) if cfg else False
        self._dllm_processed_tokens: int = 0
        self._dllm_input_tokens: int = 0
        self._dllm_decode_steps: int = 0
        # histogram of (decoded position - rightmost decoded position before the step), shifted by block length
        self._dllm_offset_hist = np.zeros((2 * self.dllm_block_length + 1, ), dtype=np.int64)
        # processed (pre-eviction) tokens split into [decoded this step, masked and still masked, already decoded]
        self._dllm_proc_breakdown = np.zeros((3, ), dtype=np.int64)
        self._last_processing_indices: Optional[torch.Tensor] = None

    @property
    def dllm_mask(self):
        start = self.num_history_ids
        end = start + self._num_token_ids
        return self.history_dllm_mask._token_ids[start:end]

    @property
    def num_valid_ids(self):
        return self._num_valid_ids

    @property
    def generated_ids(self) -> np.ndarray:
        end = self.num_valid_ids
        start = end - self.num_new_tokens
        return self.history_cache._token_ids[start:end]

    @property
    def all_dllm_mask(self):
        return self.history_dllm_mask._token_ids[:self.num_all_ids]

    @property
    def dllm_block_length(self):
        return self._strategy.block_size

    @property
    def dllm_mask_token(self):
        return self._strategy.dllm_mask_token

    @property
    def delayed_cache_enabled(self) -> bool:
        return self._delayed_cache_state is not None

    def _reset_delayed_cache_state(self):
        self._last_conf = None
        self._refeeds.fill(0)
        if self._delayed_cache_state is not None:
            self._delayed_cache_state.reset()
        if self._focus_state is not None:
            self._focus_state.reset()
        if self._focus_carry_avg and self._focus_steps > 0:
            self._focus_block_start_avg = self._focus_token_sum / self._focus_steps
        self._focus_token_sum = 0.0
        self._focus_steps = 0

    def _update_delayed_cache_state(self):
        mask = self.dllm_mask
        self._delayed_cache_state.update_from_mask(mask)

    def _window_end(self) -> int:
        """Last block position fed to the model under the sliding window."""
        last = self.dllm_block_length - 1
        if self._window_k <= 0:
            return last
        mask = self.dllm_mask
        decoded = np.flatnonzero(mask != DLLM_MASKED)
        rightmost = int(decoded[-1]) if decoded.size else -1
        end = rightmost + (self._cur_k if self._window_adapt else self._window_k)
        if self._window_conf > 0 and self._last_conf is not None:
            # SlowFast-style edge: look window_k past the farthest masked position earlier steps were already
            # fairly sure about. Positions beyond the window were never computed, so the edge has to be pushed
            # out from inside it.
            likely = np.flatnonzero((mask == DLLM_MASKED) & (self._last_conf >= self._window_conf))
            likely = likely[likely > rightmost]
            if likely.size:
                end = max(end, int(likely[-1]) + self._window_k)
            end = min(end, rightmost + self._window_k_max)
        return min(end, last)

    def update_conf(self, conf: np.ndarray):
        """Remember the confidences computed this step (0 marks positions that were not processed)."""
        if self._last_conf is None:
            self._last_conf = conf.copy()
        else:
            np.copyto(self._last_conf, conf, where=conf > 0)

    def _select_topn(self, indices: torch.Tensor) -> torch.Tensor:
        """d2Cache-style certainty prior: keep re-fed tokens and the top-n masks by conf x decoded density."""
        mask = self.dllm_mask
        idx = indices.numpy()
        is_mask = mask[idx] == DLLM_MASKED
        if int(is_mask.sum()) <= self._window_topn:
            return indices
        block = self.dllm_block_length
        # decoded positions of this block plus the (fully decoded) previous block
        decoded = np.concatenate([np.arange(-block, 0), np.flatnonzero(mask != DLLM_MASKED)])
        cand = idx[is_mask]
        dist = cand[:, None] - decoded[None, :]
        density = np.exp(-dist * dist / (2 * self._window_sigma**2)).sum(axis=1)
        score = self._last_conf[cand] * density
        keep_masks = cand[np.argsort(-score, kind='stable')[:self._window_topn]]
        keep = np.sort(np.concatenate([idx[~is_mask], keep_masks]))
        return torch.from_numpy(keep)

    def get_processing_indices(self) -> torch.Tensor:
        indices = self._delayed_cache_state.get_processing_indices()
        window_end = self._window_end()
        if window_end < self.dllm_block_length - 1:
            # Dropping a suffix keeps the slice sorted; position rightmost + 1 is always masked so it is non-empty.
            if self._fast_host:
                indices = indices[:int(np.searchsorted(indices.numpy(), window_end, side='right'))]
            else:
                indices = indices[indices <= window_end]
        if self._window_topn > 0 and self._last_conf is not None:
            indices = self._select_topn(indices)
        self._last_processing_indices = indices
        # NOTE: sparse kernels assume each per-sequence slice is sorted
        # ascending, so keep the natural torch.nonzero ordering.
        return indices

    def get_focus_info(self) -> FocusInfo:
        mask_np = self.dllm_mask
        # masked entries of this step's processing slice (get_processing_indices runs first)
        proc = self._last_processing_indices.numpy()
        keep = mask_np[proc] == DLLM_MASKED
        if self._refresh_evict_repeat:
            # a first re-feed is listed like a mask, so refresh eviction keeps it through all layers
            keep |= self._refeeds[proc] == 0
        mask_local_indices = torch.from_numpy(np.flatnonzero(keep))
        rightmost = self._focus_state.get_rightmost() if self._focus_state is not None else -1
        avg_tokens = self._focus_block_start_avg
        if self._focus_steps > 0:
            avg_tokens = self._focus_token_sum / self._focus_steps
        avg_tokens = max(avg_tokens, self._focus_min_avg)
        return FocusInfo(mask_local_indices=mask_local_indices,
                         rightmost_processed=rightmost,
                         avg_decoded_tokens=avg_tokens)

    def mark_focus_processed(self, rightmost_processed: int):
        self._focus_state.mark_processed(rightmost_processed)

    def update_focus_stats(self, decoded_tokens: int):
        if decoded_tokens <= 0:
            return
        self._focus_token_sum += int(decoded_tokens)
        self._focus_steps += 1

    def record_decode_stats(self, processed_tokens: int, input_tokens: int = None):
        self._dllm_decode_steps += 1
        self._dllm_processed_tokens += processed_tokens
        self._dllm_input_tokens += processed_tokens if input_tokens is None else input_tokens

    def adapt_window(self, prev_mask: np.ndarray, curr_mask: np.ndarray):
        """Grow the window when the last step decoded at its edge, shrink it when decodes stayed close."""
        newly = np.flatnonzero((prev_mask == DLLM_MASKED) & (curr_mask != DLLM_MASKED))
        if newly.size == 0:
            return
        decoded = np.flatnonzero(prev_mask != DLLM_MASKED)
        rightmost = int(decoded[-1]) if decoded.size else -1
        far = int(newly[-1]) - rightmost
        if far >= self._cur_k:
            self._cur_k = min(2 * self._cur_k, self._window_k_max)
        elif 2 * far <= self._cur_k:
            self._cur_k = max(self._cur_k // 2, self._window_k)

    def record_decode_offsets(self, prev_mask: np.ndarray, curr_mask: np.ndarray):
        if self._last_processing_indices is not None:
            proc = self._last_processing_indices.numpy()
            was_masked = prev_mask[proc] == DLLM_MASKED
            now_unmasked = curr_mask[proc] == DLLM_UNMASKED
            self._dllm_proc_breakdown += (int((was_masked & now_unmasked).sum()), int((was_masked & ~now_unmasked).sum()),
                                          int((~was_masked).sum()))
            self._last_processing_indices = None
        newly = np.flatnonzero((prev_mask == DLLM_MASKED) & (curr_mask == DLLM_UNMASKED))
        if newly.size == 0:
            return
        decoded = np.flatnonzero(prev_mask != DLLM_MASKED)
        rightmost = int(decoded[-1]) if decoded.size else -1
        np.add.at(self._dllm_offset_hist, newly - rightmost + self.dllm_block_length, 1)

    def get_dllm_request_stats(self) -> Dict[str, Any]:
        return dict(processed_tokens=self._dllm_processed_tokens,
                    input_tokens=self._dllm_input_tokens,
                    decode_steps=self._dllm_decode_steps,
                    offset_hist=self._dllm_offset_hist.tolist(),
                    proc_breakdown=self._dllm_proc_breakdown.tolist())

    def set_stop_pos(self, pos: int):
        dllm_block_length = self.dllm_block_length
        val = dllm_block_length - pos - 1
        self._num_valid_ids -= val
        self.num_new_tokens -= val

    def _update_token_ids_inputs(self, token_ids: np.ndarray, dllm_mask: np.ndarray):
        """Append tokens."""
        num_tokens = len(token_ids)
        dllm_block_length = self.dllm_block_length
        dllm_mask_token = self.dllm_mask_token
        new_token_ids = [token_ids]
        new_dllm_mask = [dllm_mask]

        # add uncached tokens in token_ids
        # for example, [cccc cccc uumm], the [uu] in last block is remain valid.
        num_remain_valid = self.num_valid_ids - self.num_history_ids
        if num_remain_valid != 0:
            prev_token_ids = self.valid_ids[-num_remain_valid:]
            prev_dllm_mask = np.full_like(prev_token_ids, DLLM_UNMASKED, dtype=DLLM_MASK_DTYPE)
            new_token_ids = [prev_token_ids] + new_token_ids
            new_dllm_mask = [prev_dllm_mask] + new_dllm_mask
            self.history_cache.resize(self.num_history_ids)
            self.history_dllm_mask.resize(self.num_history_ids)
            num_tokens += num_remain_valid

        # pad to align with dllm_block_length
        num_pad = (-num_tokens) % dllm_block_length
        if num_pad > 0:
            pad_ids = np.full_like(token_ids, dllm_mask_token, shape=(num_pad, ))
            pad_mask = np.full_like(dllm_mask, DLLM_MASKED, shape=(num_pad, ))
            new_token_ids += [pad_ids]
            new_dllm_mask += [pad_mask]

        token_ids = np.concatenate(new_token_ids)
        dllm_mask = np.concatenate(new_dllm_mask)

        assert len(token_ids) % dllm_block_length == 0

        self.history_cache.append(token_ids)
        self.history_dllm_mask.append(dllm_mask)
        self.output_start_pos = self._num_valid_ids + len(token_ids)
        self._num_valid_ids = self.num_history_ids + num_tokens
        self._num_token_ids = len(token_ids)
        self.num_new_tokens = 0

    def _update_token_ids_decode(self, token_ids: np.ndarray, dllm_mask: np.ndarray):
        """Update token ids for decode."""
        num_tokens = len(token_ids)
        dllm_block_length = self.dllm_block_length
        dllm_mask_token = self.dllm_mask_token
        assert num_tokens % dllm_block_length == 0
        num_history_ids = self.num_history_ids

        token_ids[dllm_mask == DLLM_MASKED] = dllm_mask_token
        self.history_cache[num_history_ids:] = token_ids
        self.history_dllm_mask[num_history_ids:] = dllm_mask

        # check if all blocks are cached
        last_mask = dllm_mask[-dllm_block_length:]
        is_unmasked = np.all(last_mask == DLLM_UNMASKED)
        is_cached = np.all(last_mask == DLLM_CACHED)

        if is_unmasked:
            num_new = dllm_block_length - self._num_valid_ids % dllm_block_length
            self._num_valid_ids += num_new
            self.num_new_tokens += num_new

        if is_cached:
            # add new block
            new_token_ids = np.full_like(token_ids, dllm_mask_token, shape=(dllm_block_length, ))
            new_dllm_mask = np.full_like(dllm_mask, DLLM_MASKED, shape=(dllm_block_length, ))
            self.history_cache.append(new_token_ids)
            self.history_dllm_mask.append(new_dllm_mask)
            self._num_history_ids += self._num_token_ids
            self._num_token_ids = dllm_block_length
            self._reset_delayed_cache_state()

    def _update_token_ids_prefill(self, token_ids: np.ndarray, dllm_mask: np.ndarray):
        """Update token ids for prefill."""
        dllm_block_length = self.dllm_block_length
        num_history_ids = self.num_history_ids

        # fill input cache
        if self.num_token_ids > dllm_block_length:
            end = self.num_token_ids - dllm_block_length
            self.history_dllm_mask[num_history_ids:end] = DLLM_CACHED
            self._num_history_ids += end
            self._num_token_ids -= end

        # decoding update
        self._update_token_ids_decode(token_ids, dllm_mask)

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         dllm_mask: Tensor = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        # update history image nums
        self._update_embeddings(embeddings)

        # update multimodals
        self._update_multimodals(multimodals)

        self.arrive_time = time.perf_counter()

        token_ids: np.ndarray = _to_ndarray(token_ids)
        if dllm_mask is None:
            dllm_mask = np.full_like(token_ids, DLLM_UNMASKED, dtype=DLLM_MASK_DTYPE)
        dllm_mask: np.ndarray = _to_ndarray(dllm_mask)

        if mode == UpdateTokenMode.INPUTS:
            self._update_token_ids_inputs(token_ids, dllm_mask)
        elif mode == UpdateTokenMode.PREFILL:
            self._update_token_ids_prefill(token_ids, dllm_mask)
        else:
            self._update_token_ids_decode(token_ids, dllm_mask)

        if model_meta is not None:
            self.model_meta = model_meta


class DLLMSequenceStrategy(SequenceStrategy):

    def __init__(self,
                 block_size: int,
                 dllm_mask_token: int,
                 enable_delayed_cache: bool = False,
                 dllm_config: DLLMConfig = None) -> None:
        self.block_size = block_size
        self.dllm_mask_token = dllm_mask_token
        self.enable_delayed_cache = enable_delayed_cache
        self.dllm_config = dllm_config
        self.track = False
        if dllm_config is not None:
            self.track = dllm_config.track

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequenceDLLM':
        """Make sequence."""
        return SchedulerSequenceDLLM(seq_id=seq_id,
                                     session=session,
                                     sampling_param=sampling_param,
                                     adapter_name=adapter_name,
                                     migration_request=migration_request,
                                     resp_cache=resp_cache,
                                     preserve_cache=preserve_cache,
                                     dllm_config=self.dllm_config)

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, model_inputs: 'ModelInputs') -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)
        dllm_mask = batched_outputs.extra_outputs.dllm_mask
        dllm_conf = getattr(batched_outputs.extra_outputs, 'dllm_conf', None)
        stop_pos = batched_outputs.stop_pos

        batch_size = len(running)
        next_token_ids = next_token_ids.view(batch_size, -1).numpy()
        dllm_mask = dllm_mask.view(batch_size, -1).numpy()
        if dllm_conf is not None:
            dllm_conf = dllm_conf.view(batch_size, -1).float().numpy()
        stop_pos = stop_pos.tolist()
        cache_delay = getattr(self.dllm_config, 'cache_delay', 'neighbor') if self.dllm_config is not None else 'neighbor'
        zero_delay = cache_delay == 'zero'
        conf_delay = cache_delay == 'conf'
        is_decoding = model_inputs.is_decoding
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL
        for idx, token in enumerate(next_token_ids):
            msg = running[idx]
            stop = stopped[idx]
            model_meta = model_metas[idx]
            mask = dllm_mask[idx]
            if msg.status != MessageStatus.RUNNING:
                continue

            prev_focus_mask = None
            if is_decoding and (msg._focus_enabled or self.track or conf_delay or msg._window_adapt):
                prev_focus_mask = msg.dllm_mask.copy()

            if is_decoding:
                if self.track:
                    processed_tokens = model_meta.get(consts.DLLM_META_PROCESSED_TOKENS)
                    input_tokens = model_meta.get(consts.DLLM_META_INPUT_TOKENS)
                    msg.record_decode_stats(processed_tokens, input_tokens)
                if self.enable_delayed_cache:
                    # Refresh delayed-cache bookkeeping using the existing mask
                    # so newly unmasked tokens still get one more iteration.
                    msg._update_delayed_cache_state()
                if msg._focus_enabled and model_meta is not None:
                    processed_rightmost = model_meta.get('focus_processed_rightmost')
                    msg.mark_focus_processed(processed_rightmost)
            if is_decoding and msg._refresh_evict_repeat and msg._last_processing_indices is not None:
                proc = msg._last_processing_indices.numpy()
                msg._refeeds[proc[prev_focus_mask[proc] != DLLM_MASKED]] += 1
            if is_decoding and dllm_conf is not None:
                # cleared again by update_token_ids if this step retired the block
                msg.update_conf(dllm_conf[idx])
            # fill token
            if is_decoding and msg._window_adapt:
                # before update_token_ids, which may swap in the next block
                msg.adapt_window(prev_focus_mask, mask)
            msg.update_token_ids(token, dllm_mask=mask, model_meta=model_meta, mode=update_mode)
            if is_decoding and zero_delay and msg.delayed_cache_enabled:
                # freeze tokens decoded this very step (their KV is the one computed from the mask input)
                msg._update_delayed_cache_state()
            if (is_decoding and conf_delay and msg.delayed_cache_enabled and dllm_conf is not None
                    and msg._last_conf is not None):
                # same block (update_token_ids clears _last_conf when it retires one): skip the refresh pass of
                # tokens the model was already near-certain about
                msg._delayed_cache_state.freeze_new(prev_focus_mask, msg.dllm_mask, dllm_conf[idx],
                                                    self.dllm_config.freeze_conf)
            if is_decoding and self.track and prev_focus_mask is not None:
                msg.record_decode_offsets(prev_focus_mask, msg.dllm_mask)
            if is_decoding and msg._focus_enabled:
                curr_mask = msg.dllm_mask
                if prev_focus_mask is not None:
                    newly_unmasked = ((prev_focus_mask == DLLM_MASKED) & (curr_mask == DLLM_UNMASKED))
                    decoded_count = int(newly_unmasked.sum())
                    msg.update_focus_stats(decoded_count)
            if stop:
                msg.set_stop_pos(stop_pos[idx])
                msg.state.finish()
