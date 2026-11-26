# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.profiler import record_function

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED


class UnmaskingProcessor:

    def __init__(self, dllm_config: DLLMConfig):
        self.dllm_config = dllm_config

    def _get_scores(self, logits: torch.Tensor, token_ids: torch.Tensor):
        """Get scores."""
        scores = logits.softmax(dim=-1)
        scores = scores.gather(-1, token_ids.unsqueeze(-1)).flatten()
        return scores

    def _get_denoise_num(self):
        """Get denoise num."""
        block_size = self.dllm_config.block_length
        denoising_steps = self.dllm_config.denoising_steps
        if denoising_steps is None:
            denoising_steps = block_size
        num = block_size // self.dllm_config.denoising_steps
        num = max(1, min(num, block_size))
        return num

    def low_confidence_static(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """static."""
        block_size = self.dllm_config.block_length
        topk = self._get_denoise_num()
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        _, indices = scores.topk(topk, dim=-1)
        dllm_unmasked = dllm_mask.scatter(-1, indices, DLLM_UNMASKED)

        is_masked = is_masked.view_as(dllm_mask)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)
        return dllm_mask.flatten()

    def _entropy_bound_select(self, logits: torch.Tensor, is_masked: torch.Tensor, block_size: int):
        """EB-Sampler selection (https://arxiv.org/abs/2505.24857, DiffusionGemma's default sampler).

        Per block, take the masks in ascending entropy order while sum(entropy) - max(entropy) <= entropy_bound
        (an upper bound on their joint mutual information). The lowest-entropy mask is always taken.
        """
        flat = logits.flatten(0, -2)
        rows = is_masked.nonzero().flatten()
        ent = torch.full((flat.size(0), ), float('inf'), dtype=torch.float32, device=flat.device)
        if rows.numel() > 0:
            logp = flat.index_select(0, rows).float().log_softmax(-1)
            ent[rows] = -(logp.exp() * logp).sum(-1)
        ent = ent.view(-1, block_size)
        sorted_ent, order = ent.sort(dim=1)
        # non-masked rows sort last with inf: inf - inf = nan compares False, so they are never taken
        take = (sorted_ent.cumsum(1) - sorted_ent) <= self.dllm_config.entropy_bound
        return torch.zeros_like(take).scatter(1, order, take) & is_masked.view(-1, block_size)

    def low_confidence_dynamic(self,
                               logits: torch.Tensor,
                               token_ids: torch.Tensor,
                               dllm_mask: torch.Tensor,
                               prev_pred: torch.Tensor = None):
        """dynamic.

        Masked positions are the processed ones (the caller marks unprocessed masks as cached). On top of the
        threshold / top-1 rule, force_first decodes the leftmost mask of every block and stable_unmask decodes the
        masks whose prediction equals prev_pred (the prediction of the last step that processed them, -1 = none)
        and, with stable_min_conf, whose confidence reaches that floor.
        """
        block_size = self.dllm_config.block_length
        threshold = self.dllm_config.confidence_threshold
        is_masked = dllm_mask == DLLM_MASKED
        if self.dllm_config.fast_unmask:
            # fused, masked rows only: same value as softmax(...).gather(...), no [rows, vocab] softmax
            from lmdeploy.pytorch.kernels.cuda.dllm_conf import token_prob
            scores = token_prob(logits.flatten(0, -2), token_ids.flatten(), is_masked)
        else:
            scores = self._get_scores(logits, token_ids)
            scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        conf = scores
        dllm_mask = dllm_mask.view(-1, block_size)
        is_masked_flat = is_masked
        is_masked = is_masked.view_as(dllm_mask)
        if self.dllm_config.entropy_bound > 0:
            unmask = self._entropy_bound_select(logits, is_masked_flat, block_size)
        elif self.dllm_config.unmask_budget > 0:
            # most confident first while sum(1 - conf) fits; non-masks (conf 0) sort last and the inf keeps them out
            vals, order = conf.float().sort(dim=1, descending=True)
            sorted_masked = is_masked.gather(1, order)
            spent = (1 - vals).masked_fill(~sorted_masked, float('inf')).cumsum(dim=1)
            take = spent <= self.dllm_config.unmask_budget
            take[:, 0] = True
            take &= sorted_masked
            unmask = torch.zeros_like(is_masked).scatter(1, order, take)
        else:
            _, indices = scores.topk(1, dim=-1)
            scores = scores.scatter(-1, indices, threshold)
            unmask = is_masked & (scores >= threshold)
        if self.dllm_config.force_first:
            # argmax gives the first True; the is_masked term drops rows without any mask
            first = is_masked.int().argmax(dim=1, keepdim=True)
            pos = torch.arange(block_size, device=dllm_mask.device)
            forced = is_masked & (pos[None, :] == first)
            if self.dllm_config.force_first_min_conf > 0:
                forced &= conf >= self.dllm_config.force_first_min_conf
            unmask |= forced
        if self.dllm_config.stable_unmask and prev_pred is not None:
            stable = is_masked & (token_ids.view(-1, block_size) == prev_pred.view(-1, block_size))
            if self.dllm_config.stable_min_conf > 0:
                stable &= conf >= self.dllm_config.stable_min_conf
            unmask |= stable
        dllm_mask[unmask] = DLLM_UNMASKED
        return dllm_mask.flatten(), conf.flatten()

    def sequential(self, dllm_mask: torch.Tensor):
        """sequential."""
        block_size = self.dllm_config.block_length
        denoise_num = self._get_denoise_num()
        dllm_mask = dllm_mask.view(-1, block_size)
        is_masked = dllm_mask == DLLM_MASKED

        # get indices
        indices = is_masked.int().argmax(dim=1)
        ranges = torch.arange(0, denoise_num, device=indices.device, dtype=indices.dtype)
        indices = indices[:, None] + ranges[None, :]
        indices = indices % block_size

        dllm_unmasked = dllm_mask.clone()
        dllm_unmasked = dllm_unmasked.scatter(-1, indices, DLLM_UNMASKED)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)

        return dllm_mask.flatten()

    @record_function('unmasking')
    def __call__(self,
                 logits: torch.Tensor,
                 input_ids: torch.Tensor,
                 token_ids: torch.Tensor,
                 dllm_mask: torch.Tensor,
                 prev_pred: torch.Tensor = None):
        """call.

        Returns (dllm_mask, token_ids, conf), where conf holds this step's confidence of every masked, processed
        position (0 elsewhere) for the dynamic strategy and is None otherwise.
        """
        strategy = self.dllm_config.unmasking_strategy
        if strategy is None:
            return dllm_mask, token_ids, None

        # reshape to [num_blocks, block_size]
        block_size = self.dllm_config.block_length
        dllm_mask = dllm_mask.unflatten(0, (-1, block_size))

        is_same = (dllm_mask == dllm_mask[:, :1]).all(dim=1)
        first_mask = dllm_mask[:, 0]

        # unmasked to cache
        is_block_unmasked = is_same & (first_mask == DLLM_UNMASKED)
        dllm_mask[is_block_unmasked] = DLLM_CACHED

        dllm_mask = dllm_mask.flatten()
        token_ids = torch.where(dllm_mask != DLLM_MASKED, input_ids, token_ids)
        conf = None
        if strategy == UnmaskingStrategy.LOW_CONFIDENCE_STATIC:
            dllm_mask = self.low_confidence_static(logits, token_ids, dllm_mask)
        elif strategy == UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC:
            dllm_mask, conf = self.low_confidence_dynamic(logits, token_ids, dllm_mask, prev_pred)
        elif strategy == UnmaskingStrategy.SEQUENTIAL:
            dllm_mask = self.sequential(dllm_mask)
        else:
            raise RuntimeError(f'strategy {strategy} not supported.')

        return dllm_mask, token_ids, conf
