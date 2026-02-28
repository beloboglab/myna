import os
import time
import math
import argparse
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from pydantic import BaseModel

from myna.common.config import load_config
from myna.lm_dataset import DPODataset
from myna.model import load_model

# 读取命令行参数
parser = argparse.ArgumentParser(description="Myna DPO")
parser.add_argument("--config_path", type=str, default="configs/dpo.yaml", help="YAML config path")
args = parser.parse_args()


def logits_to_log_probs(logits, labels):
    """logits: (B, L, V), labels: (B, L) -> (B, L)"""
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1):
    """DPO loss. ref/policy_log_probs: (B, L), mask: (B, L). Batch 前半为 chosen，后半为 rejected。"""
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[: batch_size // 2]
    reject_ref = ref_log_probs[batch_size // 2 :]
    chosen_pi = policy_log_probs[: batch_size // 2]
    reject_pi = policy_log_probs[batch_size // 2 :]
    pi_logratios = chosen_pi - reject_pi
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


class TrainConfig(BaseModel):
    # training
    dtype: str = "bf16" # Choose between ['no', 'fp8', 'fp16', 'bf16']
    learning_rate: float = 4e-8
    batch_size: int = 32
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    epochs: int = 1

    # dpo
    beta: float = 0.1

    # model
    base_model_path: str = "./final/sft/myna_25M"
    ref_model_path: str = "./final/sft/myna_25M"


    # logging
    use_swanlab: bool = True
    swanlab_project: str = "Myna-DPO"
    log_interval: int = 10

    # save
    save_dir: str = "/root/autodl-tmp/myna/dpo/checkpoints"
    save_weight: str = "dpo"
    save_interval: int = 50000
    tokenizer_path: str = "./final/dpo/myna_25M"
    model_path: str = "./final/dpo/myna_25M"

    # data
    data_path: str = "./datasets/dpo.jsonl"
    max_seq_len: int = 340
    num_workers: int = 8


train_config = TrainConfig(**load_config(args.config_path))

# 设置随机种子
set_seed(42)

# 初始化 Accelerator（配置 mixed precision）
accelerator = Accelerator(
    mixed_precision=train_config.dtype,
    gradient_accumulation_steps=train_config.accumulation_steps
)

# 初始化 swanlab
swanlab = None
if train_config.use_swanlab and accelerator.is_main_process:
    try:
        import swanlab
        swanlab.init(
            project=train_config.swanlab_project,
            experiment_name=f"{train_config.save_weight}_epochs{train_config.epochs}_bs{train_config.batch_size}_lr{train_config.learning_rate}",
            config=train_config.model_dump()
        )
    except ImportError:
        accelerator.print("Warning: swanlab not installed, skipping swanlab logging")
        swanlab = None


# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(train_config.base_model_path)
model = load_model(model_path=train_config.base_model_path)
ref_model = load_model(model_path=train_config.ref_model_path)
ref_model.eval()
ref_model.requires_grad_(False)

# 初始化 optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

# 加载数据
train_dataset = DPODataset(train_config.data_path, tokenizer, max_length=train_config.max_seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
    num_workers=train_config.num_workers,
    pin_memory=True
)

# 先 prepare dataloader，使多卡时 len 为每进程的迭代数，再计算 total_steps
train_loader = accelerator.prepare(train_loader)
len_dataloader = len(train_loader)
num_update_steps_per_epoch = math.ceil(len_dataloader / accelerator.gradient_accumulation_steps)
total_steps = num_update_steps_per_epoch * train_config.epochs

warmup_steps = int(total_steps * 0.1)  # 10% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


model, ref_model, optimizer, scheduler = accelerator.prepare(model, ref_model, optimizer, scheduler)

# 训练循环
global_step = 0
optimizer_step = 0
last_log_optimizer_step = 0

for epoch in range(train_config.epochs):
    model.train()
    epoch_start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        global_step += 1
        # DPO: batch 为 dict，拼成 chosen+rejected 大 batch
        x_chosen = batch["x_chosen"].to(accelerator.device)
        x_rejected = batch["x_rejected"].to(accelerator.device)
        y_chosen = batch["y_chosen"].to(accelerator.device)
        y_rejected = batch["y_rejected"].to(accelerator.device)
        mask_chosen = batch["mask_chosen"].to(accelerator.device)
        mask_rejected = batch["mask_rejected"].to(accelerator.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                with torch.no_grad():
                    ref_outputs = ref_model(x)
                    ref_log_probs = logits_to_log_probs(ref_outputs.logits, y)
                outputs = model(x)
                policy_log_probs = logits_to_log_probs(outputs.logits, y)
                loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=train_config.beta)

            accelerator.backward(loss)

            # 梯度裁剪和优化器更新（只在累积完成后执行）
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                # 日志输出（每 log_interval 次 optimizer 步打一次）
                if optimizer_step > 0 and optimizer_step % train_config.log_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed_time = time.time() - epoch_start_time
                    steps_per_sec = train_config.log_interval / elapsed_time if elapsed_time > 0 else 0
                    remaining_steps = total_steps - optimizer_step
                    estimated_remaining = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    log_msg = (
                        f"Epoch [{epoch+1}/{train_config.epochs}] | "
                        f"Step [{optimizer_step}/{total_steps}] | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Speed: {steps_per_sec:.2f} steps/s | "
                        f"Estimated remaining: {estimated_remaining:.0f}s"
                    )
                    accelerator.print(log_msg)

                    if swanlab:
                        swanlab.log({
                            "loss": loss.item(),
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                            "step": optimizer_step,
                        })

                    epoch_start_time = time.time()

                # 保存 checkpoint
                if optimizer_step > 0 and optimizer_step % train_config.save_interval == 0:
                    if accelerator.is_main_process:
                        checkpoint_path = os.path.join(train_config.save_dir, f"{train_config.save_weight}_step{optimizer_step}")
                        accelerator.save_state(checkpoint_path)
                        accelerator.print(f"Saved checkpoint to {checkpoint_path}")

# 11. 训练结束，保存最终模型
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(train_config.model_path)
    tokenizer.save_pretrained(train_config.model_path)
    accelerator.print(f"Training completed! Final model saved to {train_config.model_path}")

if swanlab:
    swanlab.finish()
