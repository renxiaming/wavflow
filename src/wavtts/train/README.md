# Training

> **💡 Note:** The training pipeline is consistent with [F5-TTS](https://github.com/SWivid/F5-TTS). If you encounter any issues during data preparation or training, we highly recommend checking the original F5-TTS repository's documentation and issue tracker first.

## 0. Prerequisites

Ensure FFmpeg is installed on your system. You can verify your installation by running:

```bash
ffmpeg -version
```

*(If FFmpeg is not found, please install it using your system's package manager, or ensure you have an alternative audio backend available.)*

## 1. Prepare Dataset

We provide example data processing scripts for several open-source datasets. You can adapt these scripts for your own data or customize the `Dataset` class in `src/wavtts/model/dataset.py`.

### A. Pre-built Dataset Scripts
Before running these scripts, please download the corresponding dataset and update the data paths within the script.

```bash
# Prepare the Emilia dataset (Primary dataset for WavTTS)
python src/wavtts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/wavtts/train/datasets/prepare_wenetspeech4tts.py

# Prepare the LibriTTS dataset
python src/wavtts/train/datasets/prepare_libritts.py

# Prepare the LJSpeech dataset
python src/wavtts/train/datasets/prepare_ljspeech.py
```

### B. Custom Dataset via Metadata (CSV)

If you are using a custom dataset, you can format it with a `metadata.csv` file. For detailed guidance, see [this discussion thread (#57)](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029).

```bash
python src/wavtts/train/datasets/prepare_csv_wavs.py
```

## 2. Training & Fine-Tuning

Once your dataset is ready, you can launch the training process using Hugging Face Accelerate.

### A. Training from Scratch / Pre-training

```bash
# 1. Setup accelerate config (e.g., multi-GPU DDP, mixed precision)
# This generates a config file at ~/.cache/huggingface/accelerate/default_config.yaml
accelerate config

# 2. Launch training
# YAML configuration files are located under the src/wavtts/configs/ directory.
accelerate launch src/wavtts/train/train.py --config-name WavTTS.yaml

# Example with inline overrides:
accelerate launch --mixed_precision=bf16 src/wavtts/train/train.py \
  --config-name WavTTS.yaml \
  ++datasets.batch_size_per_gpu=19200
```

### B. Fine-Tuning
For community best practices on fine-tuning, please refer to the [Finetuning Discussion Board (#57)](https://github.com/SWivid/F5-TTS/discussions/57).

The `use_ema = True` might be harmful for early-stage fine-tuned checkpoints. Because the model undergoes very few updates initially, the EMA weights remain heavily dominated by the pre-trained weights, masking your fine-tuning progress. Try turning it off by setting `load_model(..., use_ema=False)` during inference to see if it yields better results.

*(Optional: If you use TensorBoard for logging, ensure it is installed via `pip install tensorboard`.)*

## 4. Token-to-Wave Flow-GRPO

The self-prefix Flow-GRPO setup uses
`configs/WavTTS_xpred_spkconcat_flow_grpo.yaml`. Each JSONL row contributes only
`utt`, `prompt_wav`, and `prompt_token`; paired target fields are ignored. For every update, one
complete utterance is selected in a reproducibly shuffled order, a frame-aligned prefix is sampled
from 0-30%, and the remaining suffix is generated. PPO log-probability, clipping, and KL are all
masked to that suffix.

```bash
bash src/wavtts/train/run_grpo_token_xpred.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --group-size 24 \
  --batch-size-frames 4000
```

The official original Flow-GRPO SD3 configuration uses `use_cfg: true`.
The trainer defaults to full-parameter fine-tuning (`use_lora: false`); LoRA
with rank 32 and alpha 64 remains available as an opt-in. The defaults are
`use_cfg: true` and `use_lora: false`. The paper does not state a CFG switch;
`--no-cfg` selects the later no-CFG Flow-GRPO-Fast setting. Use `--use-lora` to
opt into LoRA. When CFG is disabled, the trainer forces
`cfg_strength` to `1.0` so rollout and replay use the same conditional policy.

`--group-size` is the number of stochastic completions for one utterance across all GPUs and must
be divisible by the GPU count. `--batch-size-frames` is a per-GPU rollout/replay microbatch budget;
it controls memory without changing the GRPO group used for reward normalization. Self-prefix mode
uses the complete source wav for SIM and the JSONL `prompt_text` transcript for WER. Rollout audio
is kept by default under `output_dir/rollouts/update_XXXXXXXX/`; one record per
generated wav is appended to the single file `output_dir/rollouts/rollout_metrics.jsonl` with `sim`,
`wer`, group-level `sim_min`/`sim_max`, reward, advantage, prompt metadata, and the wav path. Use
`--discard-rollouts` only when disk usage matters.


## 3. Logging with Weights & Biases (W&B)

By default, the training script does **not** log metrics to W&B unless you are authenticated. A local `wandb/` directory will be created in the path where you run the training script.

To enable W&B logging, choose one of the following methods:

**Method 1: Interactive Login**
Log in via the CLI (Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)):
```bash
wandb login
```

**Method 2: Environment Variable**
Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize) and export it:
```bash
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
```

**Method 3: Offline Mode**
If your training server lacks internet access and you want to log metrics locally for later syncing:
```bash
export WANDB_MODE=offline
```
