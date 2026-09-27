# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from wavtts.model import CFM, Trainer
from wavtts.model.dataset import load_dataset
from wavtts.model.utils import get_tokenizer


os.chdir(str(files("wavtts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("wavtts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"wavtts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    cfm_kwargs = getattr(model_cfg.model, "cfm", {}) or {}
    if not isinstance(cfm_kwargs, dict):
        cfm_kwargs = OmegaConf.to_container(cfm_kwargs, resolve=True)
    # The frozen WavLM teacher is owned by Trainer, while the small trainable
    # alignment head remains part of CFM and is checkpointed with the model.
    repa_teacher_layer = cfm_kwargs.pop("repa_teacher_layer", None)
    repa_teacher_kwargs = {
        "enabled": bool(cfm_kwargs.get("use_repa_loss", False))
        and float(cfm_kwargs.get("repa_loss_weight", 0.0)) > 0.0,
        "model_name": cfm_kwargs.pop("repa_teacher_model", "microsoft/wavlm-base-plus"),
    }
    # By default WavLMTeacher uses its final last_hidden_state. Keep an
    # explicit integer override for future intermediate-layer ablations.
    if repa_teacher_layer is not None:
        repa_teacher_kwargs["layer"] = int(repa_teacher_layer)
    speaker_embedding_kwargs = getattr(model_cfg.model, "speaker_embedding", {}) or {}
    if not isinstance(speaker_embedding_kwargs, dict):
        speaker_embedding_kwargs = OmegaConf.to_container(speaker_embedding_kwargs, resolve=True)
    condition_type = str(getattr(model_arc, "condition_type", "text"))

    exp_name = model_cfg.ckpts.exp_name
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, wav_frame_len=model_cfg.model.waveform.wav_frame_len),
        waveform_kwargs=model_cfg.model.waveform,
        vocab_char_map=vocab_char_map,
        **cfm_kwargs,
    )
    
    save_dir = model_cfg.ckpts.save_dir
    if os.path.isabs(save_dir):
        checkpoint_path = save_dir
    else:
        checkpoint_path = str(files("wavtts").joinpath(f"../../{save_dir}"))

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_entity=getattr(model_cfg.ckpts, "wandb_entity", None),
        wandb_project=getattr(model_cfg.ckpts, "wandb_project", "WavTTS"),
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        speaker_embedding_kwargs=speaker_embedding_kwargs,
        repa_teacher_kwargs=repa_teacher_kwargs,
    )

    train_dataset = load_dataset(
        model_cfg.datasets.name,
        tokenizer,
        dataset_type=getattr(model_cfg.datasets, "dataset_type", "CustomDataset"),
        waveform_kwargs=model_cfg.model.waveform,
        lance_path=getattr(model_cfg.datasets, "lance_path", None),
        text_index_path=getattr(model_cfg.datasets, "text_index_path", None),
        text_jsonl_path=getattr(model_cfg.datasets, "text_jsonl_path", None),
        polyphone=getattr(model_cfg.datasets, "polyphone", True),
        load_speech_tokens=getattr(model_cfg.datasets, "load_speech_tokens", False) or condition_type == "speech_token",
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
