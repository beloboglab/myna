import os
import time
import math
import argparse
import torch
import torch.nn as nn

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from pydantic import BaseModel

from myna.common.config import load_config
from myna.lm_dataset import SFTDataset
from myna.model import load_model


class TrainConfig(BaseModel):
    # training
    dtype: str = "bf16"
    learning_rate: float = 1e-6
    batch_size: int = 8
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    epochs: int = 1

    # logging
    use_swanlab: bool = True
    swanlab_project: str = "Myna-Reasoning"
    log_interval: int = 100

    # save
    save_dir: str = "/root/autodl-tmp/myna/checkpoints"
    save_weight: str = "reason"
    save_interval: int = 50000
    tokenizer_path: str = "./final/myna_25M"
    model_path: str = "./final/myna_25M"
    from_weight: str = "dpo"  # 基于哪个权重训练，如 dpo / full_sft
    from_resume: bool = False  # 是否从 resume checkpoint 续训

    # data
    data_path: str = "./datasets/r1_mix_1024.jsonl"
    max_seq_len: int = 720
    num_workers: int = 8

    # 推理蒸馏：特殊 token 上的 loss 权重
    special_token_loss_weight: float = 10.0


def train_epoch(
    epoch,
    train_loader,
    model,
    tokenizer,
    optimizer,
    scheduler,
    accelerator,
    config: TrainConfig,
    total_steps,
    swanlab=None,
    start_step=0,
):
    """单 epoch 训练，对 <think>/</think>/<answer></answer> 位置做 loss 加权。"""
    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    start_of_think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_of_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    start_of_answer_ids = tokenizer("<answer>", add_special_tokens=False).input_ids
    end_of_answer_ids = tokenizer("</answer>", add_special_tokens=False).input_ids
    special_ids = (
        start_of_think_ids
        + end_of_think_ids
        + start_of_answer_ids
        + end_of_answer_ids
    )
    special_ids_tensor = torch.tensor(special_ids, device=accelerator.device)

    epoch_start_time = time.time()
    optimizer_step = start_step

    for step, (input_ids, labels) in enumerate(train_loader):
        with accelerator.accumulate(model):
            with accelerator.autocast():
                res = model(input_ids)
                shift_logits = res.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_per_token = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.size())

                loss_mask = (shift_labels != -100).float()
                sp_ids = torch.isin(
                    shift_labels.view(-1),
                    special_ids_tensor,
                )
                loss_mask_flat = loss_mask.view(-1)
                loss_mask_sum = loss_mask_flat.sum()
                if loss_mask_sum > 0:
                    loss_mask_flat = loss_mask_flat.clone()
                    loss_mask_flat[sp_ids] = config.special_token_loss_weight
                    loss_mask = loss_mask_flat.view(shift_labels.size())
                    logits_loss = (loss_per_token * loss_mask).sum() / loss_mask_sum
                else:
                    logits_loss = loss_per_token.sum() * 0.0

                loss = logits_loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

        if accelerator.sync_gradients:
            if optimizer_step % config.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start_time
                steps_per_sec = config.log_interval / elapsed if elapsed > 0 else 0
                eta_min = (total_steps - optimizer_step) / steps_per_sec / 60 if steps_per_sec > 0 else 0
                msg = (
                    f"Epoch [{epoch+1}/{config.epochs}] Step [{optimizer_step}/{total_steps}] | "
                    f"loss: {loss.item():.4f} | lr: {current_lr:.2e} | eta: {eta_min:.1f}min"
                )
                accelerator.print(msg)
                if swanlab:
                    swanlab.log({
                        "loss": loss.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch + 1,
                        "step": optimizer_step,
                    })
                epoch_start_time = time.time()

            if optimizer_step % config.save_interval == 0 and accelerator.is_main_process:
                ckp_path = os.path.join(config.save_dir, f"{config.save_weight}_step{optimizer_step}")
                accelerator.save_state(ckp_path)
                accelerator.print(f"Saved checkpoint to {ckp_path}")

    return optimizer_step


def main():
    parser = argparse.ArgumentParser(description="Myna Reasoning Distillation")
    parser.add_argument("--config_path", type=str, default="configs/reason.yaml", help="YAML config path")
    args = parser.parse_args()

    config = TrainConfig(**load_config(args.config_path))
    set_seed(42)

    accelerator = Accelerator(
        mixed_precision=config.dtype,
        gradient_accumulation_steps=config.accumulation_steps,
    )

    swanlab = None
    if config.use_swanlab and accelerator.is_main_process:
        try:
            import swanlab
            swanlab.init(
                project=config.swanlab_project,
                experiment_name=f"{config.save_weight}_epochs{config.epochs}_bs{config.batch_size}_lr{config.learning_rate}",
                config=config.model_dump(),
            )
        except ImportError:
            accelerator.print("Warning: swanlab not installed, skipping logging")
            swanlab = None

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    # 从已有权重加载：优先 from_weight 目录下的权重
    from_path = config.model_path
    if config.from_weight and config.from_weight not in ("none", ""):
        cand = os.path.join(config.save_dir, config.from_weight)
        if os.path.isdir(cand):
            from_path = cand
        else:
            # 可能是前缀名，如 dpo -> save_dir/dpo_step50000
            for name in os.listdir(config.save_dir or "."):
                if name.startswith(config.from_weight + "_step"):
                    from_path = os.path.join(config.save_dir, name)
                    break
    model = load_model(model_path=from_path)

    train_dataset = SFTDataset(config.data_path, tokenizer, max_length=config.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    train_loader = accelerator.prepare(train_loader)
    len_dataloader = len(train_loader)
    num_update_steps_per_epoch = math.ceil(len_dataloader / accelerator.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * config.epochs

    warmup_steps = int(total_steps * 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    start_epoch = 0
    start_optimizer_step = 0
    if config.from_resume:
        resume_dir = os.path.join(config.save_dir, config.save_weight + "_step")
        # 找最新 step 目录
        best_step = None
        if os.path.isdir(config.save_dir):
            for name in os.listdir(config.save_dir):
                if name.startswith(config.save_weight + "_step") and name.replace(config.save_weight + "_step", "").isdigit():
                    step = int(name.replace(config.save_weight + "_step", ""))
                    if best_step is None or step > best_step:
                        best_step = step
            if best_step is not None:
                resume_path = os.path.join(config.save_dir, f"{config.save_weight}_step{best_step}")
                accelerator.load_state(resume_path)
                start_optimizer_step = best_step
                accelerator.print(f"Resumed from {resume_path}, step {best_step}")

    global_optimizer_step = start_optimizer_step
    for epoch in range(start_epoch, config.epochs):
        global_optimizer_step = train_epoch(
            epoch,
            train_loader,
            model,
            tokenizer,
            optimizer,
            scheduler,
            accelerator,
            config,
            total_steps,
            swanlab=swanlab,
            start_step=global_optimizer_step,
        )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(config.model_path)
        tokenizer.save_pretrained(config.model_path)
        accelerator.print(f"Training completed. Model saved to {config.model_path}")

    if swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()
