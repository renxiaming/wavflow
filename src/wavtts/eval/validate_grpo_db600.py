"""Reproducible, speaker-diverse DB600 comparison against the SFT checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = Path('/home/A02_tmpdata1/hq_2700h_instruct_data/instruct_data/biaobei_600h/tts_data_0515')
SIM_ROOT = Path('/home/A02_tmpdata0/wenhaoli/code/seed_eval')
SIM_PYTHON = '/home/A03_tmpdata0/minconda3/envs/eval_sim/bin/python'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def identity(row):
    parts = Path(row['prompt_wav']).parts
    parts = parts[parts.index('prompt') + 1:]
    db = re.match(r'中文[男女]声DB-\d+', parts[0])
    # DB voice IDs remain one identity across emotion/language/age subdirectories.
    speaker = db[0] if db else '/'.join(parts[:2])
    if '女声' in speaker or re.search(r'(?:[/-]|^)F\d|_F$', speaker):
        gender = 'F'
    elif '男声' in speaker or re.search(r'(?:[/-]|^)M\d|_M$', speaker):
        gender = 'M'
    else:
        gender = 'unknown'
    family = 'DB' if db else parts[0]
    return speaker, gender, family, RAW_ROOT.joinpath(*parts)


def validation_path(out, num_samples):
    return out / f'validation{num_samples}.jsonl'


def prepare(out, seed, num_samples):
    source = ROOT / 'portable_wavtts_grpo/data/db600/28w_db600.jsonl'
    groups = defaultdict(list)
    labels = {}
    for line in source.open(encoding='utf-8'):
        row = json.loads(line)
        speaker, gender, family, prompt = identity(row)
        if not (75 <= len(row['prompt_token']) <= 225 and 50 <= len(row['gen_token']) <= 200):
            continue
        if not re.search(r'[\u4e00-\u9fff]', row['text']):
            continue
        if len(groups[speaker]) >= 64:
            continue
        groups[speaker].append((row, prompt))
        labels[speaker] = (gender, family)
    rng = random.Random(seed)
    candidates = defaultdict(list)
    for speaker, options in groups.items():
        rng.shuffle(options)
        for row, prompt in options:
            if not prompt.is_file():
                continue
            info = sf.info(str(prompt))
            if not 3 <= info.duration <= 10:
                continue
            candidates[speaker].append((row, prompt, info.duration))
            if len(candidates[speaker]) == 2:
                break
    families = defaultdict(list)
    for speaker in candidates:
        families[labels[speaker][1]].append(speaker)
    for speakers in families.values():
        rng.shuffle(speakers)
    family_order = sorted(families)
    rng.shuffle(family_order)
    speaker_order = []
    while any(families.values()):
        for family in family_order:
            if families[family]:
                speaker_order.append(families[family].pop())
    # Use every available speaker at most once before adding a second utterance.
    selected = [(speaker, 0) for speaker in speaker_order[:num_samples]]
    repeats_needed = num_samples - len(selected)
    repeat_speakers = [speaker for speaker in speaker_order if len(candidates[speaker]) > 1]
    rng.shuffle(repeat_speakers)
    if repeats_needed > len(repeat_speakers):
        raise RuntimeError(
            f'Requested {num_samples} rows, but only {len(speaker_order)} distinct speakers and '
            f'{len(repeat_speakers)} speakers with a second eligible utterance are accessible.'
        )
    selected.extend((speaker, 1) for speaker in repeat_speakers[:repeats_needed])
    rng.shuffle(selected)
    rows, details = [], []
    width = max(2, len(str(num_samples)))
    for index, (speaker, candidate_index) in enumerate(selected):
        original, prompt, duration = candidates[speaker][candidate_index]
        utt = f'db600_sample{index + 1:0{width}d}'
        row = dict(utt=utt, source_utt=original['utt'], speaker_id=speaker,
                   prompt_wav=str(prompt), prompt_text=original['prompt_text'],
                   prompt_token=original['prompt_token'], target_text=original['text'],
                   target_token=original['gen_token'], language='zh')
        rows.append(row)
        details.append(dict(utt=utt, source_utt=original['utt'], speaker_id=speaker,
                            gender=labels[speaker][0], prompt_wav=str(prompt),
                            prompt_seconds=duration, target_seconds=len(row['target_token']) / 25,
                            generation_seed=seed + index * 1_000_003,
                            prompt_text=row['prompt_text'], target_text=row['target_text']))
    distinct_speakers = len({r['speaker_id'] for r in rows})
    assert len(rows) == num_samples
    jsonl = validation_path(out, num_samples)
    jsonl.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    genders = Counter(labels[speaker][0] for speaker, _ in selected)
    write_json(out / 'selection.json', dict(source=str(source), audio_root=str(RAW_ROOT),
        available_candidate_speakers=len(candidates), n=num_samples,
        distinct_speakers=distinct_speakers, repeated_speaker_rows=num_samples - distinct_speakers,
        female=genders['F'], male=genders['M'], unknown_gender=genders['unknown'], seed=seed,
        selection_rule='One row per speaker first; second rows only after all accessible speakers are used.',
        samples=details,
        target_duration='Inferred from fixed target tokens at 25 Hz; identical for all checkpoints.'))
    return jsonl


def checkpoints(out):
    directory = ROOT / 'result/grpo_flowtts_v2'
    entries = [dict(name='sft_model_80000', path=str(ROOT / 'portable_wavtts_grpo/ckpts/model_80000.pt'))]
    excluded, signatures = [], {}
    for path in sorted(directory.glob('model_*.pt'), key=lambda p: int(re.search(r'\d+', p.stem)[0]) if re.search(r'\d+', p.stem) else 10**9):
        try:
            with zipfile.ZipFile(path) as archive:
                signature = tuple((item.filename.split('/', 1)[-1], item.CRC, item.file_size) for item in archive.infolist())
        except zipfile.BadZipFile:
            excluded.append(dict(path=str(path), reason='Incomplete/corrupt PyTorch ZIP archive'))
            continue
        if signature in signatures:
            excluded.append(dict(path=str(path), reason='Duplicate archive contents', same_as=signatures[signature]))
            continue
        signatures[signature] = str(path)
        entries.append(dict(name=path.stem, path=str(path)))
    write_json(out / 'checkpoints.json', dict(evaluated=entries, excluded=excluded))
    return entries


def infer_case(entry, gpu, out, seed, jsonl, num_samples):
    name = entry['name']
    case = out / name
    case.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / 'src'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    args = [sys.executable, '-m', 'wavtts.eval.infer_grpo', '--model-cfg',
            str(ROOT / 'src/wavtts/configs/WavTTS_xpred_spkconcat_flow_grpo.yaml'),
            '--ckpt', entry['path'], '--jsonl', str(jsonl),
            '--output-dir', str(case), '--device', 'cuda:0', '--sampler', 'ode',
            '--steps', '20', '--cfg-strength', '1.5', '--timestep-mapping', 'power',
            '--timestep-power', '2', '--shift', '3', '--model-dtype', 'fp32',
            '--compute-dtype', 'bf16', '--data-mode', 'paired', '--seed', str(seed)]
    with (out / 'logs' / f'{name}_infer.log').open('w') as log:
        subprocess.run(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = [json.loads(l) for l in (case / 'manifest.jsonl').open()]
    if len(manifest) != num_samples or not all(Path(r['gen_wav']).is_file() for r in manifest):
        raise RuntimeError(f'Incomplete inference for {name}')
    print(f'INFER COMPLETE {name}: {num_samples}/{num_samples}', flush=True)


def sim_case(entry, gpu, out, jsonl, num_samples):
    name = entry['name']
    case = out / name
    sim = out / 'sim' / name
    sim.mkdir(parents=True, exist_ok=True)
    # The same verification2.py, arguments and WavLM-large checkpoint as run_sim.sh.
    args = [SIM_PYTHON, str(SIM_ROOT / 'verification2.py'), '--wavs', str(case),
            '--target_spk', str(RAW_ROOT), '--metalst', str(case / 'meta_zh.lst'),
            '--output', str(sim), '--device', 'cuda:0']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with (out / 'logs' / f'{name}_sim.log').open('w') as log:
        subprocess.run(args, env=env, cwd=SIM_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    scores = read_scores(sim / 'total_similarity.txt')
    expected = {json.loads(l)['utt'] for l in jsonl.open()}
    if set(scores) != expected:
        raise RuntimeError(f'Incomplete SIM for {name}: {len(scores)}/{num_samples}')
    print(f'SIM COMPLETE {name}: mean={np.mean(list(scores.values())):.5f}', flush=True)


def read_scores(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    return {line.rsplit(': ', 1)[0]: float(line.rsplit(': ', 1)[1]) for line in lines[1:] if line.strip()}


def report(entries, out):
    selection = json.loads((out / 'selection.json').read_text())
    samples = selection['samples']
    num_samples = len(samples)
    scores = {e['name']: read_scores(out / 'sim' / e['name'] / 'total_similarity.txt') for e in entries}
    base = np.array([scores['sft_model_80000'][s['utt']] for s in samples])
    rng = np.random.default_rng(666)
    speaker_groups = defaultdict(list)
    for index, sample in enumerate(samples):
        speaker_groups[sample['speaker_id']].append(index)
    cluster_indices = [np.array(indices) for indices in speaker_groups.values()]
    cluster_resamples = rng.integers(
        0, len(cluster_indices), size=(20000, len(cluster_indices))
    )
    summaries = []
    for entry in entries:
        values = np.array([scores[entry['name']][s['utt']] for s in samples])
        delta = values - base
        cluster_sums = np.array([delta[indices].sum() for indices in cluster_indices])
        cluster_counts = np.array([len(indices) for indices in cluster_indices])
        bootstrap_delta = (
            cluster_sums[cluster_resamples].sum(axis=1)
            / cluster_counts[cluster_resamples].sum(axis=1)
        )
        lo, hi = np.quantile(bootstrap_delta, [0.025, 0.975])
        summaries.append(dict(checkpoint=entry['name'], n=num_samples, mean_sim=float(values.mean()),
            mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
            improved=int((delta > 0).sum()), worse=int((delta < 0).sum()),
            ties=int((delta == 0).sum()), paired_ci95_low=float(lo), paired_ci95_high=float(hi)))
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    with (out / 'per_utterance.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['utt', 'speaker_id', 'prompt_text', 'target_text']
        fields += [e['name'] for e in entries]
        fields += [e['name'] + '_delta' for e in entries[1:]]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {k: sample[k] for k in fields[:4]}
            row.update({e['name']: scores[e['name']][sample['utt']] for e in entries})
            row.update({e['name'] + '_delta': scores[e['name']][sample['utt']] - scores['sft_model_80000'][sample['utt']] for e in entries[1:]})
            writer.writerow(row)
    write_json(out / 'summary.json', summaries)
    best = max(summaries[1:], key=lambda s: s['mean_sim'])
    lines = [f"# DB600：{num_samples} 条 / {selection['distinct_speakers']} 个不同说话人 SIM 验证", '',
        f"本次平均 SIM 最高的是 {best['checkpoint']}：{best['mean_sim']:.5f}；相对原模型增加 {best['mean_delta']:.5f}，{best['improved']}/{num_samples} 条改善。",
        '各 checkpoint 存在波动，没有随训练步数持续上升；这批样本有提升迹象，仍需独立数据和多个生成种子确认。', '',
        f"{num_samples} 条、{selection['distinct_speakers']} 个独立说话人；固定每条生成种子。完整 prompt 作参考，仅生成的 target 段参与 SIM。",
        'ODE，20 步，CFG=1.5，power=2，shift=3，FP32 参数 / BF16 前向。',
        'SIM 使用 seed_eval/run_sim.sh 对应的 verification2.py 和 WavLM-large 权重；原始结果精度为四位小数。',
        '', '| Checkpoint | 平均 SIM | 相对 SFT | 提升条数 | 说话人聚类 bootstrap 95% CI |',
        '|---|---:|---:|---:|---|']
    for s in summaries:
        lines.append(f"| {s['checkpoint']} | {s['mean_sim']:.5f} | {s['mean_delta']:+.5f} | {s['improved']}/{num_samples} | [{s['paired_ci95_low']:+.5f}, {s['paired_ci95_high']:+.5f}] |")
    lines += ['', '选择最佳 checkpoint 后仍需要独立验证集确认，不能把多个 checkpoint 的最佳结果视为独立显著性结论。',
        '本次仅评测 SIM，不据此判断 MOS 或 WER。v2 checkpoint 记录的 GRPO 训练源为 Seed JSONL；这不能证明 db600 未参与过原始 SFT 预训练。',
        'Checkpoint 对比评估的是整套微调后的效果；不能单凭这次结果把提升全部归因于 GRPO reward，训练中的其他损失也可能贡献提升。',
        '', '跳过的 checkpoint：']
    for e in json.loads((out / 'checkpoints.json').read_text())['excluded']:
        lines.append(f"- {Path(e['path']).name}: {e['reason']}" + (f"；等同 {Path(e['same_as']).name}" if 'same_as' in e else ''))
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--gpus', default='0,2,4,6')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--stage', choices=['prepare', 'run', 'sim', 'report'], default='run')
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    jsonl = validation_path(out, args.num_samples)
    if not jsonl.exists():
        prepare(out, args.seed, args.num_samples)
    entries = checkpoints(out)
    settings = dict(seed=args.seed, sampler='ode', steps=20, cfg_strength=1.5,
        timestep_mapping='power', timestep_power=2, shift=3, model_dtype='fp32', compute_dtype='bf16',
        num_samples=args.num_samples, jsonl_sha256=hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        sim_script=str(SIM_ROOT / 'verification2.py'),
        sim_checkpoint=str(SIM_ROOT / 'ckpts/wavlm_large_finetune.pth'))
    write_json(out / 'settings.json', settings)
    selection = json.loads((out / 'selection.json').read_text())
    print(f"PREPARED {out}: {selection['n']} rows / {selection['distinct_speakers']} speakers; "
          f'{len(entries)} checkpoints including SFT', flush=True)
    if args.stage == 'prepare':
        return
    gpus = args.gpus.split(',')
    def worker(gpu, jobs):
        for entry in jobs:
            if args.stage == 'run':
                infer_case(entry, gpu, out, args.seed, jsonl, args.num_samples)
            if args.stage in ('run', 'sim'):
                sim_case(entry, gpu, out, jsonl, args.num_samples)
    if args.stage != 'report':
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu, entries[i::len(gpus)]) for i, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
    report(entries, out)


if __name__ == '__main__':
    main()
