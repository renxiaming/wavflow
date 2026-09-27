# Portable WavTTS GRPO Package

This directory is a trimmed package for GRPO training/inference.

Included:
- GRPO source code under `src/wavtts`
- portable GRPO config under `configs/`
- SFT init checkpoint under `ckpts/model_80000.pt`
- tokenizer vocab under `data/Emilia_Lance_pinyin/`
- speaker embedding ONNX under `pretrain_models/`
- Chinese SIM/WER reward assets under `third_party/sim_wer_test/`
- rewritten JSONL under `data/db600/28w_db600.jsonl`
- audio copy manifest under `tools/audio_manifest.tsv`

Not included by default:
- full training audio under `data/audio/`
- English Whisper reward model
- old experiment checkpoints/logs

To copy all audio into this package on the source machine:

```bash
cd portable_wavtts_grpo
python tools/copy_audio_manifest.py --root .
```

To run after audio is present:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH}"
bash scripts/run_grpo_token_xpred.sh --gpus 0,1
```

This local package defaults to:

```bash
source /home/A02_tmpdata2/xmren/miniconda3/bin/activate wavtts
```

If the target server needs a different conda activation:

```bash
export GRPO_CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export GRPO_CONDA_ENV=wavtts
bash scripts/run_grpo_token_xpred.sh --gpus 0,1
```
