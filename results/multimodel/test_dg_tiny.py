"""CPU checks of exps/multimodel/dg_modes.py on a tiny random DiffusionGemma (transformers >= 5.8).

1. run_decoder on a canvas prefix equals the model's own forward on that shorter canvas (window correctness).
2. With nothing cut (window K >= canvas, FOCUS budget >= canvas) generate is bit-identical to the stock generate.
3. Every mode runs on a padded batch; positions without fresh logits are never accepted; window never shrinks.
Usage: python test_dg_tiny.py
"""
import copy
import os
import sys

import torch
from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'exps', 'multimodel'))
import dg_modes  # noqa: E402

torch.manual_seed(0)
CANVAS, VOCAB = 16, 384
text = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, global_head_dim=32, num_global_key_value_heads=1, sliding_window=64,
            num_experts=4, top_k_experts=2, moe_intermediate_size=32, pad_token_id=0, eos_token_id=1, bos_token_id=2,
            layer_types=['sliding_attention', 'sliding_attention', 'full_attention', 'full_attention'])
vision = dict(model_type='gemma4_vision', hidden_size=32, intermediate_size=32, num_hidden_layers=1,
              num_attention_heads=2, num_key_value_heads=2, head_dim=16, global_head_dim=16)
cfg = DiffusionGemmaConfig(text_config=text, vision_config=vision, canvas_length=CANVAS, eos_token_id=[1],
                           boi_token_id=VOCAB - 3, eoi_token_id=VOCAB - 2, image_token_id=VOCAB - 1)
base_model = DiffusionGemmaForBlockDiffusion(cfg).eval()
# a sharper random model so the samplers accept a few tokens per step
with torch.no_grad():
    base_model.lm_head.weight.mul_(8.0)
print('params', sum(p.numel() for p in base_model.parameters()))

GEN = dict(max_new_tokens=2 * CANVAS, max_denoising_steps=12, t_min=0.4, t_max=0.8, stability_threshold=1,
           confidence_threshold=0.005, sampler_config=dg_modes.gen.EntropyBoundSamplerConfig(entropy_bound=0.5),
           pad_token_id=0, eos_token_id=[1])
# padded batch: row 0 shorter (left padding)
ids = torch.randint(3, VOCAB - 4, (3, 9))
attn = torch.ones_like(ids)
ids[0, :3] = 0
attn[0, :3] = 0


def generate(mode=None, seed=1, batch=ids, mask=attn, check=False):
    model = copy.copy(base_model)  # shallow: shares weights, own instance attributes for the patches
    stats = {}
    if mode is not None:
        dg_modes.install(model, mode, stats)
        if check:
            orig = dg_modes.ModeSampler.accept_canvas

            def checked(self, *a, **k):
                prev_window = self.window
                out = orig(self, *a, **k)
                if self.valid is not None:
                    assert not (self.accepted_token_mask & ~self.valid).any(), 'accepted a position without logits'
                assert self.window >= prev_window
                return out

            dg_modes.ModeSampler.accept_canvas = checked
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(batch, attention_mask=mask, **GEN)
    if mode is not None and check:
        dg_modes.ModeSampler.accept_canvas = orig
    return out.sequences, stats


# 1. window prefix == model forward on the shorter canvas
with torch.no_grad():
    model = base_model
    cache = DynamicCache(config=cfg.get_text_config(decoder=True))
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    enc_mask = model.model.encoder.create_masks_for_generate(config=cfg, inputs_embeds=torch.empty(3, 9, 0),
                                                             attention_mask=attn.bool(), past_key_values=cache,
                                                             position_ids=pos)
    model.model.encoder(input_ids=ids, attention_mask=enc_mask, past_key_values=cache, position_ids=pos)
    canvas = torch.randint(0, VOCAB, (3, CANVAS))
    sc = torch.randn(3, CANVAS, VOCAB).to(model.dtype)
    dec_pos = torch.arange(9, 9 + CANVAS).unsqueeze(0)
    dmask2d = torch.nn.functional.pad(attn.bool(), (0, CANVAS), value=True)
    full_map = model.model.decoder.create_diffusion_decoder_attention_mask(
        config=cfg, inputs_embeds=canvas.unsqueeze(-1), past_key_values=cache, decoder_attention_mask=dmask2d)
    assert any(m is not None for m in full_map.values()), 'padded batch should produce 4D masks'
    ref_full = model(decoder_input_ids=canvas, self_conditioning_logits=sc, decoder_attention_mask=full_map,
                     past_key_values=cache, decoder_position_ids=dec_pos).logits
    got_full, _ = dg_modes.run_decoder(model, canvas, sc, full_map, cache, dec_pos)
    assert torch.equal(got_full, ref_full), (got_full - ref_full).abs().max()
    for w in (5, 11):
        keep1 = torch.arange(w).expand(3, w)
        got, idx = dg_modes.run_decoder(model, canvas, sc, full_map, cache, dec_pos, keep1=keep1)
        wmask2d = dmask2d[:, :9 + w]
        wmap = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=cfg, inputs_embeds=canvas[:, :w].unsqueeze(-1), past_key_values=cache,
            decoder_attention_mask=wmask2d)
        ref = model(decoder_input_ids=canvas[:, :w], self_conditioning_logits=sc[:, :w], decoder_attention_mask=wmap,
                    past_key_values=cache, decoder_position_ids=dec_pos[:, :w]).logits
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (w, (got - ref).abs().max())
        assert torch.equal(idx, keep1)
    # eviction keeping everything == no eviction
    keep2 = torch.arange(CANVAS).expand(3, CANVAS)
    got_k, _ = dg_modes.run_decoder(model, canvas, sc, full_map, cache, dec_pos, keep2=keep2, evict_layer=2)
    assert torch.allclose(got_k, ref_full, atol=1e-5, rtol=1e-5)
print('run_decoder: OK (full, window prefixes, no-op eviction)')

# 2. nothing cut -> identical to stock generate
ref, _ = generate(None)
for mode in (dg_modes.Mode(), dg_modes.Mode(window_k=10 * CANVAS), dg_modes.Mode(focus=True, focus_min_keep=CANVAS),
             dg_modes.Mode(window_k=10 * CANVAS, focus=True, focus_min_keep=CANVAS)):
    got, stats = generate(mode)
    assert torch.equal(got, ref), mode
print('no-op modes == stock generate: OK', tuple(ref.shape))

# 3. real modes on the padded batch
modes = {
    'window4': dg_modes.Mode(window_k=4),
    'focus': dg_modes.Mode(focus=True, focus_min_keep=2),
    'focus_age2': dg_modes.Mode(focus=True, focus_min_keep=2, focus_max_skip=2),
    'window4_focus': dg_modes.Mode(window_k=4, focus=True, focus_min_keep=2),
    'stable0.5': dg_modes.Mode(stable_min_conf=0.5),
    'stable0.0': dg_modes.Mode(stable_min_conf=0.0),
    'force_first': dg_modes.Mode(force_first=True),
    'thr0.5': dg_modes.Mode(rule='thr', threshold=0.5),
    'window4_stable_ff': dg_modes.Mode(window_k=4, stable_min_conf=0.5, force_first=True),
}
for name, mode in modes.items():
    got, stats = generate(mode, check=True)
    assert got.shape[0] == 3
    ratio = stats['decoder_tokens_deep'] / max(stats['decoder_tokens_l01'], 1)
    print(f'  {name:18s} ok  steps={stats["decoder_steps"]:3d}  deep/l01 tokens={ratio:.2f}  '
          f'same as base={bool(torch.equal(got, ref))}')
    if 'window' in name or name == 'focus':
        assert stats['decoder_tokens_deep'] < stats['decoder_steps'] * 3 * CANVAS, 'nothing was cut'
print('modes: OK')
