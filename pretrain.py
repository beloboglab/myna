import os
import time
import math
import argparse
import torch

from dataclasses import dataclass
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from myna.lm_dataset import PretrainDataset
from myna.model import load_model
from dataclasses import asdict


@dataclass
class PretrainConfig:
    # training
    dtype: str = "bf16" # Choose between ['no', 'fp8', 'fp16', 'bf16']
    learning_rate: float = 5e-4
    batch_size: int = 64
    accumulation_steps: int = 8
    grad_clip: float = 1.0
    epochs: int = 1

    # logging
    use_swanlab: bool = True
    swanlab_project: str = "Myna-Pretrain"
    log_interval: int = 1

    # save
    save_dir: str = "/root/autodl-tmp/myna/checkpoints/pretrain"
    save_weight: str = "pretrain"
    save_interval: int = 5000
    tokenizer_path: str = "./final/myna_25M"
    model_path: str = "./final/myna_25M"

    # data
    data_path: str = "./datasets/pretrain_hq.jsonl"
    max_seq_len: int = 340
    num_workers: int = 8
    


# 设置随机种子
set_seed(42)

# 初始化 Accelerator（配置 mixed precision）
accelerator = Accelerator(
    mixed_precision=PretrainConfig.dtype,
    gradient_accumulation_steps=PretrainConfig.accumulation_steps
)

# 初始化 swanlab
swanlab = None
if PretrainConfig.use_swanlab and accelerator.is_main_process:
    try:
        import swanlab
        swanlab.init(
            project=PretrainConfig.swanlab_project,
            experiment_name=f"{PretrainConfig.save_weight}_epochs{PretrainConfig.epochs}_bs{PretrainConfig.batch_size}_lr{PretrainConfig.learning_rate}",
            config=asdict(PretrainConfig())
        )
    except ImportError:
        accelerator.print("Warning: swanlab not installed, skipping swanlab logging")
        swanlab = None


# 5. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(PretrainConfig.tokenizer_path)
model = load_model()

# 6. 初始化 optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=PretrainConfig.learning_rate)

# 7. 加载数据
pretrain_dataset = PretrainDataset(PretrainConfig.data_path, tokenizer, max_length=PretrainConfig.max_seq_len)
pretrain_loader = DataLoader(
    pretrain_dataset,
    batch_size=PretrainConfig.batch_size,
    shuffle=True,
    num_workers=PretrainConfig.num_workers,
    pin_memory=True
)

# 8. 先 prepare dataloader，使多卡时 len 为每进程的迭代数，再计算 total_steps
pretrain_loader = accelerator.prepare(pretrain_loader)
len_dataloader = len(pretrain_loader)
num_update_steps_per_epoch = math.ceil(len_dataloader / accelerator.gradient_accumulation_steps)
total_steps = num_update_steps_per_epoch * PretrainConfig.epochs

warmup_steps = int(total_steps * 0.1)  # 10% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 9. 准备模型、优化器、scheduler
model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

# 10. 训练循环
global_step = 0
optimizer_step = 0
last_log_optimizer_step = 0

for epoch in range(PretrainConfig.epochs):
    model.train()
    epoch_start_time = time.time()
    
    for step, (input_ids, labels) in enumerate(pretrain_loader):
        global_step += 1
        with accelerator.accumulate(model):
            with accelerator.autocast():
                res = model(input_ids, labels=labels)
                loss = res.loss

            accelerator.backward(loss)

            # 梯度裁剪和优化器更新（只在累积完成后执行）
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), PretrainConfig.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

        

                # 日志输出（每 log_interval 次 optimizer 步打一次）
                if optimizer_step > 0 and optimizer_step % PretrainConfig.log_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed_time = time.time() - epoch_start_time
                    steps_per_sec = PretrainConfig.log_interval / elapsed_time if elapsed_time > 0 else 0
                    remaining_steps = total_steps - optimizer_step
                    estimated_remaining = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    log_msg = (
                        f"Epoch [{epoch+1}/{PretrainConfig.epochs}] | "
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
                if optimizer_step > 0 and optimizer_step % PretrainConfig.save_interval == 0:
                    if accelerator.is_main_process:
                        checkpoint_path = os.path.join(PretrainConfig.save_dir, f"{PretrainConfig.save_weight}_step{optimizer_step}")
                        accelerator.save_state(checkpoint_path)
                        accelerator.print(f"Saved checkpoint to {checkpoint_path}")

# 11. 训练结束，保存最终模型
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    final_model_path = PretrainConfig.model_path    
    unwrapped_model.save_pretrained(PretrainConfig.model_path)
    tokenizer.save_pretrained(PretrainConfig.model_path)
    accelerator.print(f"Training completed! Final model saved to {PretrainConfig.model_path}")

if swanlab:
    swanlab.finish()
