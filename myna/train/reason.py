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

# 读取命令行参数
parser = argparse.ArgumentParser(description="Myna Reasoning Distillation")
parser.add_argument("--config_path", type=str, default="configs/reason.yaml", help="YAML config path")
args = parser.parse_args()

class TrainConfig(BaseModel):
    # training
    dtype: str = "bf16"
    learning_rate: float = 1e-6
    batch_size: int = 8
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    epochs: int = 1

    # reasoning model
    base_model_path: str = "final/myna_25M_dpo"
    special_token_loss_weight: float = 10.0

    # logging
    use_swanlab: bool = True
    swanlab_project: str = "Myna-Reasoning"
    log_interval: int = 100

    # save
    save_dir: str = "/root/autodl-tmp/myna/checkpoints"
    save_weight: str = "reason"
    save_interval: int = 50000
    tokenizer_path: str = "final/myna_25M_reason"
    model_path: str = "final/myna_25M_reason"

    # data
    data_path: str = "./datasets/r1_mix_1024.jsonl"
    max_seq_len: int = 720
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

# 5. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(train_config.base_model_path)
model = load_model(model_path=train_config.base_model_path)

# 6. 初始化 optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

# 7. 加载数据
train_dataset = SFTDataset(train_config.data_path, tokenizer, max_length=train_config.max_seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
    num_workers=train_config.num_workers,
    pin_memory=True
)

# 8. 先 prepare dataloader，使多卡时 len 为每进程的迭代数，再计算 total_steps
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

# 9. 准备模型、优化器、scheduler
model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)


# 推理蒸馏：特殊 token 的 id 与 loss 权重
loss_fct = nn.CrossEntropyLoss(reduction="none")
start_of_think_ids = tokenizer("<think>", add_special_tokens=False).input_ids
end_of_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
start_of_answer_ids = tokenizer("<answer>", add_special_tokens=False).input_ids
end_of_answer_ids = tokenizer("</answer>", add_special_tokens=False).input_ids
special_ids = start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids
special_ids_tensor = torch.tensor(special_ids, device=accelerator.device)

# 10. 训练循环
for epoch in range(start_epoch, train_config.epochs):
    model.train()
    epoch_start_time = time.time()

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
                    loss_mask_flat[sp_ids] = train_config.special_token_loss_weight
                    loss_mask = loss_mask_flat.view(shift_labels.size())
                    logits_loss = (loss_per_token * loss_mask).sum() / loss_mask_sum
                else:
                    logits_loss = loss_per_token.sum() * 0.0

                loss = logits_loss

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
