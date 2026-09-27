# Inference

WavTTS supports zero-shot TTS from a reference audio prompt via either command-line or script-based inference.

> **💡 Tip for Reference Audio:**
> For optimal results, use a short reference clip, preferably under 12 seconds. We also recommend leaving a brief silence at the end of the clip to avoid truncating the prompt mid-word.

The pretrained checkpoint and matching vocabulary are available on [🤗 Hugging Face](https://huggingface.co/worstchan/WavTTS) and the checkpoint will be downloaded automatically when using the default configuration. To use a local checkpoint, specify its path with the `ckpt_file` parameter.


## CLI Inference

> **Note:** `wavtts_infer-cli` is an alias for `python src/wavtts/infer/infer_cli.py`. You can use either command interchangeably.

CLI inference can be run either with direct arguments or with a TOML configuration file. Command-line arguments always override values defined in the TOML config.

### A. Direct Arguments

Run inference by passing the required arguments directly:

```bash
wavtts_infer-cli \
  --model WavTTS \
  --ref_audio "infer/examples/basic_ref_en.wav" \
  --ref_text "Some call me nature, others call me mother nature." \
  --gen_text "The text you want WavTTS to synthesize."
```
*(Optional: Use `--model_cfg` instead of `--model` to provide an explicit YAML model config path.)*

### B. TOML Configuration

For reproducible runs, we recommend storing inference settings in a `.toml` file. If no config is provided, `wavtts_infer-cli` uses the default example config at `src/wavtts/infer/examples/basic.toml`.

```bash
# Use the default example config
wavtts_infer-cli -c src/wavtts/infer/examples/basic.toml

# Use a custom config with optional argument overrides
wavtts_infer-cli -c custom.toml --gen_text "Override text here."
```

**Example `custom.toml`:**
```toml
# Model settings
model = "WavTTS"
ckpt_file = "" # Leave empty to use the Hugging Face default
vocab_file = "infer/examples/vocab.txt"

# Prompt and generation
ref_audio = "infer/examples/basic_ref_en.wav"
ref_text = "Some call me nature, others call me mother nature."
gen_text = "The text you want WavTTS to synthesize."

# Output settings
output_dir = "output"
output_file = "infer_cli_basic.wav"
remove_silence = false

# Generation hyperparameters
nfe_step = 50
cfg_strength = 1.5
timestep_mapping = "power"
timestep_power = 2.0
shift = 3.0
speed = 1.0
```

## Token-to-Wave Inference

`infer.sh` is the main batch ODE inference entry point for SFT checkpoints. It also accepts GRPO
LoRA checkpoints because `infer_xpred_flow.py` automatically restores their recorded base model.

```bash
bash src/wavtts/infer/infer.sh \
  --ckpt-dir /path/to/checkpoints \
  --model-cfg /path/to/config.yaml \
  --ckpt-steps last \
  --jsonl /path/to/prompt_target.jsonl \
  --output-dir results/token_to_wave \
  --gpus 0 \
  --nfe-step 50 \
  --cfg-strength 1.5 \
  --shift 3.0
```

`infer_dit9.sh` is retained as a compatibility wrapper around `infer.sh` for the previous dit9
experiment defaults.

## GRPO Inference

Use `infer_grpo.sh` when comparing final ODE inference against the stochastic SDE rollout used for
GRPO exploration. It supports multiple GPUs, multiple processes per GPU, and a steps/CFG grid.
Final model quality should be evaluated with `--sampler ode`.

```bash
bash src/wavtts/infer/infer_grpo.sh \
  --ckpt /path/to/grpo/model_last.pt \
  --sampler ode \
  --steps 50 \
  --max-rows 10
```

Run a multi-GPU grid. `--nfe-step` is an alias for `--steps`; both mean the number of denoising
function evaluations for the selected sampler.

```bash
bash src/wavtts/infer/infer_grpo.sh \
  --ckpt /path/to/grpo/model_last.pt \
  --output-dir results/grpo_grid \
  --gpus 0,1,2,3 \
  --procs-per-gpu 1 \
  --sampler ode \
  --nfe-step 20,40,50 \
  --cfg-strength 1.0,1.5,2.0
```

Each combination is written to a directory such as
`results/grpo_grid/model_last/ode_steps40_cfg1p5/`. The JSONL is split across workers and shard
manifests are merged after all workers finish.

To evaluate several checkpoints from one training directory and run SeedTTS-ZH speaker similarity
after every inference case:

```bash
bash src/wavtts/infer/infer_grpo.sh \
  --ckpt-dir exp/nar_wav_tts_grpo_fp32——2 \
  --ckpt-steps 3200,4200,5000 \
  --steps 30,50 \
  --cfg-strength 1.5,2.0 \
  --gpus 0,1,2,3 \
  --sim-device cuda:7
```

`--ckpt-steps` selects files such as `model_3200.pt`; `--steps` controls inference NFE. Use
`--ckpt-steps all` to select all numbered checkpoints. Outputs are separated as follows:

```text
results/grpo_infer/model_3200/ode_steps30_cfg1p5/
results/sim/zh/model_3200/ode_steps30_cfg1p5/total_similarity.txt
```

SIM evaluation is enabled by default. Pass `--skip-sim` when only generated audio is needed.

The grid is resumable. Re-running the same command skips a case when its merged manifest contains
all expected rows and every generated wav is present. For a partially completed case, workers keep
existing non-empty wav files and generate only the missing files while rebuilding manifests. SIM is
run only when `total_similarity.txt` is missing or does not contain one score per metadata row.

To inspect training-style stochastic samples, change the sampler and use the training noise scale:

```bash
bash src/wavtts/infer/infer_grpo.sh \
  --ckpt /path/to/grpo/model_last.pt \
  --sampler sde \
  --steps 10 \
  --noise-scale 0.05 \
  --max-rows 10
```

Both scripts default to FP32 model parameters and BF16 forward autocast.

For checkpoints trained with the self-prefix Flow-GRPO config, use the paired wrapper. It reads only
the original prompt utterance and speech tokens, applies the same deterministic 0-30% prefix policy
to every checkpoint/configuration, and automatically runs SeedTTS-ZH SIM after each case.

```bash
bash src/wavtts/infer/infer_flow_grpo.sh \
  --ckpt-dir exp/nar_wav_tts_flow_grpo_xpred_prefix \
  --ckpt-steps 100,200,300 \
  --nfe-step 20,30,50 \
  --cfg-strength 1.0,1.5,2.0 \
  --gpus 0,1,2,3 \
  --sim-device cuda:7
```

The prefix seed is included in each case directory name, so configurations are directly comparable
and a different prefix policy cannot accidentally reuse stale audio or SIM results.

> **💡 Tip:**
> If you notice obvious background noise in the synthesized speech, try lowering the `shift` value to mitigate it (e.g., set `shift = 1.0`).
