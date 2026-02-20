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

parser = argparse.ArgumentParser(description="MiniMind Pretraining")
parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="混合精度类型")
parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
parser.add_argument("--data_path", type=str, default="./datasets/pretrain_hq.jsonl", help="预训练数据路径")
parser.add_argument("--use_swanlab", action="store_true", default=True, help="是否使用swanlab（默认启用）")
parser.add_argument("--no_swanlab", dest="use_swanlab", action="store_false", help="禁用swanlab")
parser.add_argument("--swanlab_project", type=str, default="MiniMind-Pretrain", help="swanlab项目名")
args = parser.parse_args()

@dataclass
class PretrainConfig:
    # training
    dtype: str = "bf16" # Choose between ['no', 'fp8', 'fp16', 'bf16']

    # logging
    use_swanlab: bool = True
    swanlab_project: str = "Myna-Pretrain"

    # save
    save_dir: str = "/root/autodl-tmp/myna/checkpoints"
    save_weight: str = "pretrain"
    save_interval: int = 5000
    tokenizer_path: str = "./final/myna_25M"
    model_path: str = "./final/myna_25M"


# 设置随机种子
set_seed(42)

# 初始化 Accelerator（配置 mixed precision）
accelerator = Accelerator(
    mixed_precision=PretrainConfig.dtype,
    gradient_accumulation_steps=args.accumulation_steps
)

# 初始化 swanlab
swanlab = None
if PretrainConfig.use_swanlab and accelerator.is_main_process:
    try:
        import swanlab
        swanlab.init(
            project=PretrainConfig.swanlab_project,
            experiment_name=f"{args.save_weight}_epochs{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}",
            config=vars(args)
        )
    except ImportError:
        accelerator.print("Warning: swanlab not installed, skipping swanlab logging")
        swanlab = None


# 5. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(PretrainConfig.tokenizer_path)
model = load_model()

# 6. 初始化 optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# 7. 加载数据
pretrain_dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
pretrain_loader = DataLoader(
    pretrain_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)

# 8. 先 prepare dataloader，使多卡时 len 为每进程的迭代数，再计算 total_steps
pretrain_loader = accelerator.prepare(pretrain_loader)
len_dataloader = len(pretrain_loader)
num_update_steps_per_epoch = math.ceil(len_dataloader / accelerator.gradient_accumulation_steps)
total_steps = num_update_steps_per_epoch * args.epochs

warmup_steps = int(total_steps * 0.1)  # 10% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 9. 准备模型、优化器、scheduler（dataloader 已在上面 prepare）
model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

# 10. 训练循环
global_step = 0
optimizer_step = 0
last_log_optimizer_step = 0

for epoch in range(args.epochs):
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
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

        

                # 日志输出（每 log_interval 次 optimizer 步打一次）
                if optimizer_step > 0 and optimizer_step % args.log_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed_time = time.time() - epoch_start_time
                    steps_per_sec = args.log_interval / elapsed_time if elapsed_time > 0 else 0
                    remaining_steps = total_steps - optimizer_step
                    estimated_remaining = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    log_msg = (
                        f"Epoch [{epoch+1}/{args.epochs}] | "
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

            # 保存 checkpoint（按 optimizer 步数间隔）
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
