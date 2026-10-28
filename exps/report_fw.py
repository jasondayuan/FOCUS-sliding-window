"""FOCUS + window (the combined method): slack sweeps on GSM8K and HumanEval against FOCUS, delayed cache + window
without eviction, and vanilla.

The FOCUS + window sessions re-run FOCUS (and delayed cache + window) as same-session anchors, so they are pooled
with the earlier repeats: every point is normalised to FOCUS slack 1.0 of its own session (see report2.summarize).
"""
import glob
import os

from report2 import HERE, OUT, SERIES_FW, matched, plot, summarize, table


def dirs(*patterns):
    return sorted(d for p in patterns for d in glob.glob(os.path.join(HERE, p)) if os.path.isdir(d))


def main():
    os.makedirs(OUT, exist_ok=True)
    gsm_fw = dirs('results_fw_rep*')
    gsm, n_gsm = summarize(dirs('results_slack', 'results_rep[0-9]') + gsm_fw)
    he_fw = dirs('results_he_fw_rep*')
    he, n_he = summarize(dirs('results_he_x4') + he_fw)
    plot(gsm, n_gsm, 'GSM8K accuracy (%)', 'pareto_gsm8k_fw',
         f'GSM8K (n=1319): FOCUS + window vs. its parts ({len(gsm_fw)} FOCUS + window sessions)',
         series=SERIES_FW, labelled=('fw4_s', 'focus_s'))
    plot(he, len(he_fw) + 1, 'HumanEval pass@1 (%), mean of 4 passes', 'pareto_humaneval_fw',
         f'HumanEval (164 x 4 passes): FOCUS + window vs. its parts ({len(he_fw)} FOCUS + window sessions)',
         series=SERIES_FW, labelled=('fw4_s', 'focus_s'))
    md = [f'## GSM8K: FOCUS + window, {len(gsm_fw)} sessions (baselines pooled over {n_gsm} sessions)\n', table(gsm),
          '\n### FOCUS + window k=4 at matched accuracy (GSM8K)\n', matched(gsm, prefix='fw4_s'),
          '\n### Delayed cache + window k=4, no eviction, at matched accuracy (GSM8K, for comparison)\n',
          matched(gsm, prefix='dc_w4_s'),
          f'\n## HumanEval, 4 passes per run: FOCUS + window, {len(he_fw)} sessions\n', table(he),
          '\n### FOCUS + window k=4 at matched accuracy (HumanEval)\n', matched(he, prefix='fw4_s'),
          '\n### Delayed cache + window k=4 at matched accuracy (HumanEval, for comparison)\n',
          matched(he, prefix='dc_w4_s')]
    text = '\n'.join(md) + '\n'
    with open(os.path.join(OUT, 'fw_results.md'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
