"""Tables for the multi-model sweep. Usage: python summarize.py [results/multimodel]  (prints markdown)"""
import glob
import json
import os
import sys

base = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def table(title, rows, cols, cell, extra_hdr, extra):
    print(f'\n### {title}\n')
    print('| mode | ' + ' | '.join(f'bs={c} tok/s' for c in cols) + ' | ' + ' | '.join(extra_hdr) + ' |')
    print('|' + '---|' * (1 + len(cols) + len(extra_hdr)))
    for r in rows:
        vals = [cell(r, c) for c in cols]
        if all(v == '' for v in vals):
            continue
        print(f'| {r} | ' + ' | '.join(vals) + ' | ' + ' | '.join(extra(r)) + ' |')


# SDAR-1.7B (LMDeploy)
sd = os.path.join(base, 'sdar')
SD_MODES = ['vanilla', 'focus', 'fw', 'fw_st05', 'fw_ff', 'fw_ff05', 'fw_eb01', 'fw_eb05']
SD_BS = [1, 8, 32, 128]


def sd_cell(m, bs):
    r = load(os.path.join(sd, f'{m}_bs{bs}.json'))
    return f"{r['out_tok_per_s']:.0f}" if r else ''


def sd_extra(m):
    r = load(os.path.join(sd, f'{m}_bs128.json'))
    if not r:
        return ['', '', '']
    return [f"{100 * r['accuracy']:.1f}", f"{r['tokens_per_step']:.2f}", f"{r['proc_per_out_tok']:.2f}"]


table('SDAR-1.7B-Chat-b32, LMDeploy (eager), GSM8K', SD_MODES, SD_BS, sd_cell,
      ['acc % (1319, bs=128)', 'tok/step', 'proc/out tok'], sd_extra)

# I-DLM-8B (SGLang ISD)
idl = os.path.join(base, 'idlm')
ID_CFG = ['n2', 'n4', 'n8', 'n4_agree05', 'n4_top5', 'n4_noverify']
ID_BS = [1, 8, 32, 64]


def id_cell(m, bs):
    r = load(os.path.join(idl, f'{m}_bs{bs}.json'))
    return f"{r['out_tok_per_s']:.0f}" if r else ''


def id_extra(m):
    r = load(os.path.join(idl, f'{m}_bs64.json'))
    if not r:
        return ['', '']
    tps = r.get('tokens_per_req_step')
    return [f"{100 * r['accuracy']:.1f}", f'{tps:.2f}' if tps else '']


table('I-DLM-8B, SGLang ISD (greedy), GSM8K', ID_CFG, ID_BS, id_cell, ['acc % (512, bs=64)', 'tok/request-step'],
      id_extra)

# DiffusionGemma-26B-A4B (transformers)
dg = os.path.join(base, 'dg')
DG_MODES = ['eb', 'eb_stable0.5', 'eb_ff', 'thr0.9', 'eb_focus', 'eb_focus_age2', 'eb_focus_window64',
            'eb_focus_window64_age2', 'eb_focus_window64_stable0.5', 'eb_focus_window64_ff']
DG_BS = [1, 4, 8]


def dg_find(m, bs):
    hits = sorted(glob.glob(os.path.join(dg, f'{m}_bs{bs}_n*.json')))
    return load(hits[-1]) if hits else None


def dg_cell(m, bs):
    r = dg_find(m, bs)
    return f"{r['out_tok_per_s']:.0f}" if r else ''


def dg_extra(m):
    r = dg_find(m, 8)
    r1 = dg_find(m, 1)
    if not r:
        return ['', '', '', '']
    deep = r['decoder_tokens_deep'] / max(r['decoder_tokens_l01'], 1)
    acc1 = f"{100 * r1['accuracy']:.0f}" if r1 else ''
    return [f"{100 * r['accuracy']:.1f}", acc1, f"{r['decoder_steps']}", f'{deep:.2f}']


table('DiffusionGemma-26B-A4B-it, transformers, GSM8K', DG_MODES, DG_BS, dg_cell,
      ['acc % (64, bs=8)', 'acc % (8, bs=1)', 'decoder steps (bs=8)', 'deep/L0-1 tokens (bs=8)'], dg_extra)
