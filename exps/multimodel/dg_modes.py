"""Decoding modes for DiffusionGemma (transformers >= 5.8, DiffusionGemmaForBlockDiffusion).

DiffusionGemma is a uniform-noise block diffusion model: a 256-token canvas starts as random tokens, every step the
decoder re-predicts the whole canvas, the entropy-bounded (EB) sampler accepts the lowest-entropy predictions and
re-randomizes everything else, and a canvas is done once its argmax is stable and its mean entropy is below 0.005.
There are no mask tokens and nothing stays decoded, so the SDAR modes are mapped as follows:

  window K   only canvas positions [0, W) go through the decoder, W = max over the batch of (frontier + K), where the
             frontier is the run of positions accepted in the last step starting at 0 (W never shrinks within a
             canvas). The first step of a canvas sees the whole canvas (like the delayed cache's first step).
  focus      FOCUS-style eviction after layer `evict_layer`: deep layers only see the tokens accepted in the last step
             plus the ceil(alpha * avg newly accepted per step) (at least min_keep) lowest-entropy noisy positions;
             the other noisy positions (random tokens nobody will accept this step) are dropped. Importance is the last
             step's entropy, not FOCUS's attention score.
  stable c   also accept positions whose argmax equals the last step's argmax with probability >= c (the argmax token
             is taken).
  force_first  also accept the leftmost processed position the rule did not accept.
  rule thr   accept probability >= threshold (at least the most confident one) instead of the EB rule.

Positions that are not processed in a step keep their stale logits (for the stopping rule and self-conditioning) and
are never accepted in that step.
"""
from dataclasses import dataclass

import torch

from transformers.models.diffusion_gemma import generation_diffusion_gemma as gen


@dataclass
class Mode:
    rule: str = 'eb'  # 'eb' (model default, uses generation_config.sampler_config.entropy_bound) or 'thr'
    threshold: float = 0.9
    stable_min_conf: float = -1.0  # < 0: off
    force_first: bool = False
    window_k: int = 0  # 0: off
    focus: bool = False
    focus_alpha: float = 1.5
    focus_min_keep: int = 16
    evict_layer: int = 2
    # > 0: a position dropped from the deep layers this many steps in a row is kept in the next one (its importance,
    # the last computed entropy, is otherwise never refreshed and it can starve)
    focus_max_skip: int = 0

    @property
    def custom_step(self):
        return self.window_k > 0 or self.focus


class ModeSampler(gen.EntropyBoundSampler):
    """EntropyBoundSampler plus the acceptance variants; with the default Mode it accepts exactly what the base does."""

    def __init__(self, base: gen.EntropyBoundSampler, mode: Mode):
        self.__dict__.update(base.__dict__)
        self.mode = mode
        self.reset()

    def reset(self):
        self.accepted_token_mask = None
        self.valid = None  # positions with fresh logits this step (None = all)
        self.prev_argmax = None
        self.last_entropy = None
        self.window = 0
        self.skip = None  # consecutive steps each position missed the deep layers
        self.new_sum = None  # newly accepted tokens summed over the steps of this canvas, per sequence
        self.steps = 0

    def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
        mode = self.mode
        valid = self.valid
        prev_accepted = self.accepted_token_mask
        plain = mode.rule == 'eb' and mode.stable_min_conf < 0 and not mode.force_first and valid is None
        if plain and not mode.focus:
            # the base sampler itself (FOCUS needs the entropies kept, so it takes the equivalent path below)
            accepted = super().accept_canvas(current_canvas, denoiser_canvas, logits, cur_step)
            self._bookkeep(prev_accepted, None, None)
            return accepted
        dist = torch.distributions.Categorical(logits=logits)
        token_entropy = dist.entropy()
        self.last_entropy = token_entropy
        ent = token_entropy if valid is None else token_entropy.masked_fill(~valid, float('inf'))
        if mode.rule == 'eb':
            sorted_ent, order = torch.sort(ent, dim=-1)
            take = (torch.cumsum(sorted_ent, dim=-1) - sorted_ent) <= self.entropy_bound  # nan (inf - inf) -> False
            accept = torch.zeros_like(take).scatter(-1, order, take)
        elif mode.rule == 'thr':
            pmax = dist.probs.max(-1).values
            if valid is not None:
                pmax = pmax.masked_fill(~valid, -1.0)
            accept = pmax >= mode.threshold
            accept |= torch.zeros_like(accept).scatter(-1, pmax.argmax(-1, keepdim=True), True)
        else:
            raise ValueError(mode.rule)
        if valid is not None:
            accept &= valid
        tokens = torch.where(accept, denoiser_canvas, current_canvas)
        argmax = logits.argmax(-1)
        if mode.stable_min_conf >= 0 and self.prev_argmax is not None:
            pmax = dist.probs.max(-1).values
            stable = (argmax == self.prev_argmax) & (pmax >= mode.stable_min_conf) & ~accept
            if valid is not None:
                stable &= valid
            tokens = torch.where(stable, argmax, tokens)
            accept |= stable
        if mode.force_first:
            cand = ~accept if valid is None else (~accept & valid)
            first = cand.int().argmax(-1, keepdim=True)
            forced = torch.zeros_like(accept).scatter(-1, first, True) & cand
            tokens = torch.where(forced, denoiser_canvas, tokens)
            accept |= forced
        self.accepted_token_mask = accept
        self._bookkeep(prev_accepted, argmax, valid)
        return tokens

    def _bookkeep(self, prev_accepted, argmax, valid):
        accept = self.accepted_token_mask
        if argmax is None or valid is None or self.prev_argmax is None:
            self.prev_argmax = argmax
        else:
            self.prev_argmax = torch.where(valid, argmax, self.prev_argmax)
        new = accept if prev_accepted is None else accept & ~prev_accepted
        n = new.sum(-1).float()
        self.new_sum = n if self.new_sum is None else self.new_sum + n
        self.steps += 1


def _subset_masks(mask_mapping, canvas_len, num):
    """4D decoder masks for a canvas subset of `num` positions (bidirectional: only padded cache keys are masked)."""
    out = {}
    for key, mask in mask_mapping.items():
        if mask is None:
            out[key] = None
            continue
        cache_cols = mask.shape[-1] - canvas_len
        row = mask[:, :, :1, :cache_cols]
        fill = torch.ones if mask.dtype == torch.bool else torch.zeros
        canvas = fill((*row.shape[:-1], num), dtype=mask.dtype, device=mask.device)
        out[key] = torch.cat([row, canvas], dim=-1).expand(mask.shape[0], mask.shape[1], num, cache_cols + num)
    return out


def run_decoder(model, canvas, self_conditioning_logits, mask_mapping, past_key_values, decoder_position_ids,
                keep1=None, keep2=None, evict_layer=2):
    """DiffusionGemmaForBlockDiffusion.forward for the decoder with a canvas subset.

    keep1 [B, M1]: canvas positions fed to the decoder (None = all). keep2 [B, M2]: indices into the keep1 subset that
    stay after layer `evict_layer` (None = all). Returns (logits [B, M, V] float32, canvas positions [B, M]).
    """
    dec = model.model.decoder
    cfg = dec.text_config
    bsz, canvas_len = canvas.shape
    idx = keep1 if keep1 is not None else torch.arange(canvas_len, device=canvas.device).expand(bsz, canvas_len)
    ids = canvas.gather(1, idx)
    inputs_embeds = dec.embed_tokens(ids)
    if self_conditioning_logits is not None:
        sc = self_conditioning_logits if keep1 is None else self_conditioning_logits.gather(
            1, idx[..., None].expand(-1, -1, self_conditioning_logits.shape[-1]))
        soft = torch.matmul(sc.softmax(dim=-1, dtype=torch.float32).to(dec.embed_tokens.weight.dtype),
                            dec.embed_tokens.weight) * dec.embed_tokens.embed_scale.to(inputs_embeds.dtype)
    else:
        soft = torch.zeros_like(inputs_embeds)
    hidden = dec.self_conditioning(inputs_embeds, soft)
    pos = decoder_position_ids.expand(bsz, -1).gather(1, idx)
    masks = mask_mapping if keep1 is None else _subset_masks(mask_mapping, canvas_len, idx.shape[1])
    pe = {lt: dec.rotary_emb(hidden, pos, lt) for lt in dec.unique_layer_types}
    for i, layer in enumerate(dec.layers[:cfg.num_hidden_layers]):
        if keep2 is not None and i == evict_layer:
            hidden = hidden.gather(1, keep2[..., None].expand(-1, -1, hidden.shape[-1]))
            idx = idx.gather(1, keep2)
            pos = pos.gather(1, keep2)
            masks = _subset_masks(mask_mapping, canvas_len, idx.shape[1])
            pe = {lt: dec.rotary_emb(hidden, pos, lt) for lt in dec.unique_layer_types}
        lt = cfg.layer_types[i]
        hidden = layer(hidden, position_embeddings=pe[lt], attention_mask=masks[lt], position_ids=pos,
                       past_key_values=past_key_values)
    hidden = dec.norm(hidden)
    logits = model.lm_head(hidden).to(torch.float32)
    cap = model.final_logit_softcapping
    logits = torch.tanh(logits / cap) * cap
    return logits, idx


def _focus_keep(sampler: ModeSampler, mode: Mode, window: int):
    """Per sequence: last step's accepted tokens + the budget of lowest-entropy noisy positions, within the window.

    Returns keep2 [B, M] (sorted canvas positions < window; every row has the same M, rows needing fewer get extra
    low-entropy positions) and the bool keep mask [B, window].
    """
    ent = sampler.last_entropy[:, :window]
    settled = sampler.accepted_token_mask[:, :window]
    avg_new = sampler.new_sum / max(sampler.steps, 1)
    budget = torch.clamp(torch.ceil(mode.focus_alpha * avg_new), min=mode.focus_min_keep).long()
    noisy_ent = ent.masked_fill(settled, float('inf'))
    rank = noisy_ent.argsort(-1).argsort(-1)  # 0 = lowest-entropy noisy position
    keep = settled | (rank < budget[:, None])
    if mode.focus_max_skip > 0 and sampler.skip is not None:
        keep |= sampler.skip[:, :window] >= mode.focus_max_skip
    num = int(keep.sum(-1).max())
    # pad rows to the same count with their next-lowest-entropy positions
    score = torch.where(keep, torch.full_like(ent, -1.0), noisy_ent.masked_fill(settled, float('inf')))
    top = score.topk(num, dim=-1, largest=False).indices
    keep2 = top.sort(-1).values
    keep_mask = torch.zeros_like(keep).scatter(-1, keep2, True)
    return keep2, keep_mask


def make_denoising_step(model, mode: Mode, stats: dict):
    """A replacement for DiffusionGemmaGenerationMixin._denoising_step implementing window / focus."""

    def step(decoder_forward, current_canvas, argmax_canvas, input_ids, decoder_position_ids, self_conditioning_logits,
             mask_mapping, past_key_values, finished_denoising, cur_step, sampler, logits_processor,
             diffusion_stopping_criteria, **model_kwargs):
        first = self_conditioning_logits is None
        if first:
            sampler.reset()
        cur_step_t = torch.tensor(cur_step, device=current_canvas.device, dtype=torch.int32)
        bsz, canvas_len = current_canvas.shape
        keep1 = keep2 = None
        window = canvas_len
        if not first:
            if mode.window_k > 0:
                acc = sampler.accepted_token_mask
                lead = torch.cumprod(acc.int(), dim=-1).sum(-1)  # leading accepted run per sequence
                sampler.window = max(sampler.window, min(canvas_len, int(lead.max()) + mode.window_k))
                window = sampler.window
                if window < canvas_len:
                    keep1 = torch.arange(window, device=current_canvas.device).expand(bsz, window)
            if mode.focus:
                keep2, _ = _focus_keep(sampler, mode, window)
                if keep2.shape[1] == window:
                    keep2 = None
        logits_sub, idx = run_decoder(model, current_canvas, self_conditioning_logits, mask_mapping, past_key_values,
                                      decoder_position_ids, keep1, keep2, mode.evict_layer)
        processed_sub = logits_processor(input_ids, logits_sub, cur_step=cur_step_t)
        if idx.shape[1] == canvas_len and keep1 is None and keep2 is None:
            processed_logits = processed_sub
            sampler.valid = None
        else:
            # stale (last step's, already temperature-scaled) logits where nothing was computed
            processed_logits = self_conditioning_logits.to(torch.float32).clone()
            processed_logits.scatter_(1, idx[..., None].expand(-1, -1, processed_sub.shape[-1]), processed_sub)
            valid = torch.zeros((bsz, canvas_len), dtype=torch.bool, device=current_canvas.device)
            sampler.valid = valid.scatter_(1, idx, True)
        proc = sampler.valid if sampler.valid is not None else torch.ones(
            (bsz, canvas_len), dtype=torch.bool, device=current_canvas.device)
        sampler.skip = torch.where(proc, 0, (sampler.skip if sampler.skip is not None else 0) + 1)
        n_proc = idx.shape[1]
        stats['decoder_tokens_l01'] = stats.get('decoder_tokens_l01', 0) + bsz * (
            keep1.shape[1] if keep1 is not None else canvas_len)
        stats['decoder_tokens_deep'] = stats.get('decoder_tokens_deep', 0) + bsz * n_proc
        stats['decoder_steps'] = stats.get('decoder_steps', 0) + 1

        probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
        vocab_size = model.config.text_config.vocab_size
        denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).squeeze(-1).view(
            bsz, canvas_len)
        new_argmax_canvas = torch.argmax(processed_logits, dim=-1)
        accepted_canvas = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, cur_step_t).clone()
        new_current_canvas = sampler.renoise_canvas(accepted_canvas, cur_step_t).clone()
        if diffusion_stopping_criteria is not None:
            if finished_denoising.any():
                new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
                new_current_canvas = torch.where(finished_denoising[:, None], current_canvas, new_current_canvas)
                processed_logits = torch.where(finished_denoising[:, None, None], self_conditioning_logits,
                                               processed_logits)
            finished_denoising |= diffusion_stopping_criteria(new_argmax_canvas, processed_logits)
        self_conditioning_logits = processed_logits.to(model.model.decoder.embed_tokens.weight.dtype)
        return new_current_canvas, new_argmax_canvas, self_conditioning_logits, finished_denoising

    return step


def install(model, mode: Mode, stats: dict):
    """Patch the model instance: mode-aware sampler, and the custom step when window / focus is on."""
    orig_prepare_sampler = type(model)._prepare_sampler

    def prepare_sampler(self, generation_config):
        return ModeSampler(orig_prepare_sampler(self, generation_config), mode)

    model._prepare_sampler = prepare_sampler.__get__(model)
    if mode.custom_step:
        model._denoising_step = make_denoising_step(model, mode, stats)
    else:
        orig_step = type(model)._denoising_step

        def counted_step(self, *args, **kwargs):
            if kwargs.get('self_conditioning_logits') is None:  # first step of a canvas
                kwargs['sampler'].reset()
            out = orig_step(self, *args, **kwargs)
            bsz, canvas_len = out[0].shape
            stats['decoder_tokens_l01'] = stats.get('decoder_tokens_l01', 0) + bsz * canvas_len
            stats['decoder_tokens_deep'] = stats.get('decoder_tokens_deep', 0) + bsz * canvas_len
            stats['decoder_steps'] = stats.get('decoder_steps', 0) + 1
            return out

        model._denoising_step = counted_step.__get__(model)
    return model

