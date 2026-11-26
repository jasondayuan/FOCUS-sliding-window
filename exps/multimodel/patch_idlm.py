"""Patch I-DLM's ISD algorithm (inference/sglang/sglang/srt/dllm/algorithm/idlm_blockN.py) in place:

  verify_min_prob  (config key, default 0): in the greedy path a speculative token is only accepted when it equals the
                   verify-pass argmax AND has probability >= this floor ("two consecutive predictions agree, > 0.5").
  stats file       every 20 forwards the counters are written to $IDLM_STATS_PATH (json), with decode_reqs =
                   request-steps, so tokens per request-step = total_tokens / decode_reqs over a run.
Usage: python patch_idlm.py <path to idlm_blockN.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:100]
    s = s.replace(old, new, 1)


rep("""        self.fast_verify: bool = config.algorithm_config.get("fast_verify", False)""",
    """        self.fast_verify: bool = config.algorithm_config.get("fast_verify", False)
        # (patch) greedy verify also needs p(spec) >= verify_min_prob
        self.verify_min_prob: float = config.algorithm_config.get("verify_min_prob", 0.0)""")
rep("""                    all_accepted_gpu = all_spec_vals_t == all_corr_tokens_gpu
""", """                    all_accepted_gpu = all_spec_vals_t == all_corr_tokens_gpu
                    if self.verify_min_prob > 0:
                        p_spec = combined_logits[:verify_count].float().softmax(-1).gather(
                            1, all_spec_vals_t.unsqueeze(1)).squeeze(1)
                        all_accepted_gpu = all_accepted_gpu & (p_spec >= self.verify_min_prob)
""")
rep("""        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
""", """        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        self._stats["decode_reqs"] = self._stats.get("decode_reqs", 0) + len(decode_bids)
        if self._stats["total_forwards"] % 20 == 0:
            import json as _json
            import os as _os
            _path = _os.environ.get("IDLM_STATS_PATH")
            if _path:
                with open(_path + ".tmp", "w") as _f:
                    _json.dump(self._stats, _f)
                _os.replace(_path + ".tmp", _path)
""")
open(p, 'w').write(s)
print('patched', p)
