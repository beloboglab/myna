import os
import re
import time
import math
import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass, asdict
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Qwen3ForCausalLM, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from myna.lm_dataset import RLAIFDataset
from myna.model import load_model


class CriticModel(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def score_with_reward_model(messages, reward_model, reward_tokenizer, device):
    """Score a conversation using Skywork reward model. Returns a scalar score."""
    conv_formatted = reward_tokenizer.apply_chat_template(messages, tokenize=False)
    if reward_tokenizer.bos_token and conv_formatted.startswith(reward_tokenizer.bos_token):
        conv_formatted = conv_formatted[len(reward_tokenizer.bos_token):]
    inputs = reward_tokenizer(conv_formatted, return_tensors="pt").to(device)
    score = reward_model(**inputs).logits[0][0].item()
    return score


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, device, reasoning=0):
    def reasoning_model_reward(rewards):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, r, re.S) for r in responses]
        matches_pattern2 = [re.match(pattern2, r, re.S) for r in responses]
        format_rewards = [0.5 if (m1 or m2) else 0.0
                          for m1, m2 in zip(matches_pattern, matches_pattern2)]
        rewards += torch.tensor(format_rewards, device=device)

        def mark_num(text):
            r = 0
            if text.count("<think>") == 1: r += 0.25
            if text.count("</think>") == 1: r += 0.25
            if text.count("<answer>") == 1: r += 0.25
            if text.count("</answer>") == 1: r += 0.25
            return r

        rewards += torch.tensor([mark_num(r) for r in responses], device=device)
        return rewards

    rewards = torch.zeros(len(responses), device=device)
    if reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            # Extract user messages from ChatML-formatted prompt (skip system for Skywork)
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()}
                        for role, content in matches if role != "system"]

            chat = messages + [{"role": "assistant", "content": response}]
            score = score_with_reward_model(chat, reward_model, reward_tokenizer, device)
            scale = 3.0
            score = max(min(score, scale), -scale)

            if reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = score_with_reward_model(chat, reward_model, reward_tokenizer, device)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        rewards += torch.tensor(reward_model_scores, device=device)

    return rewards


@dataclass
class TrainConfig:
    # training
    dtype: str = "bf16"
    learning_rate: float = 8e-8
    critic_learning_rate: float = 8e-8
    batch_size: int = 2
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    epochs: int = 1

    # ppo
    clip_epsilon: float = 0.1
    vf_coef: float = 0.5
    kl_coef: float = 0.02
    max_gen_len: int = 1536
    update_old_actor_freq: int = 4
    reasoning: int = 0  # 0=normal, 1=reasoning model

    # model
    base_model_path: str = "./final/sft/myna_25M"
    reward_model_path: str = "/root/autodl-tmp/Skywork/Skywork-Reward-V2-Qwen3-1.7B"

    # logging
    use_swanlab: bool = True
    swanlab_project: str = "Myna-PPO"
    log_interval: int = 1

    # save
    save_dir: str = "/root/autodl-tmp/myna/ppo/checkpoints"
    save_weight: str = "ppo_actor"
    save_interval: int = 10
    tokenizer_path: str = "./final/ppo/myna_25M"
    model_path: str = "./final/ppo/myna_25M"

    # data
    data_path: str = "./datasets/rlaif-mini.jsonl"
    max_seq_len: int = 66
    num_workers: int = 8


set_seed(42)

accelerator = Accelerator(
    mixed_precision=TrainConfig.dtype,
    gradient_accumulation_steps=TrainConfig.accumulation_steps
)

swanlab = None
if TrainConfig.use_swanlab and accelerator.is_main_process:
    try:
        import swanlab
        swanlab.init(
            project=TrainConfig.swanlab_project,
            experiment_name=f"{TrainConfig.save_weight}_epochs{TrainConfig.epochs}_bs{TrainConfig.batch_size}_lr{TrainConfig.learning_rate}",
            config=asdict(TrainConfig())
        )
    except ImportError:
        accelerator.print("Warning: swanlab not installed, skipping swanlab logging")
        swanlab = None

# ==================== Load models ====================
tokenizer = AutoTokenizer.from_pretrained(TrainConfig.base_model_path)

actor_model = load_model(model_path=TrainConfig.base_model_path)

old_actor_model = load_model(model_path=TrainConfig.base_model_path)
old_actor_model.eval().requires_grad_(False)

ref_model = load_model(model_path=TrainConfig.base_model_path)
ref_model.eval().requires_grad_(False)

critic_model = CriticModel.from_pretrained(TrainConfig.base_model_path)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    TrainConfig.reward_model_path, torch_dtype=torch.bfloat16, num_labels=1
)
reward_model.eval().requires_grad_(False)
reward_tokenizer = AutoTokenizer.from_pretrained(TrainConfig.reward_model_path)

# ==================== Optimizers & data ====================
actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=TrainConfig.learning_rate)
critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=TrainConfig.critic_learning_rate)

train_dataset = RLAIFDataset(TrainConfig.data_path, tokenizer, max_length=(TrainConfig.max_seq_len + TrainConfig.max_gen_len))
train_loader = DataLoader(
    train_dataset,
    batch_size=TrainConfig.batch_size,
    shuffle=True,
    num_workers=TrainConfig.num_workers,
    pin_memory=True
)

train_loader = accelerator.prepare(train_loader)
len_dataloader = len(train_loader)
num_update_steps_per_epoch = math.ceil(len_dataloader / accelerator.gradient_accumulation_steps)
total_steps = num_update_steps_per_epoch * TrainConfig.epochs

warmup_steps = int(total_steps * 0.1)
actor_scheduler = get_cosine_schedule_with_warmup(
    actor_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
critic_scheduler = get_cosine_schedule_with_warmup(
    critic_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

actor_model, critic_model, actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler = accelerator.prepare(
    actor_model, critic_model, actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler
)

old_actor_model = old_actor_model.to(accelerator.device)
ref_model = ref_model.to(accelerator.device)
reward_model = reward_model.to(accelerator.device)

# ==================== Training ====================
optimizer_step = 0

for epoch in range(TrainConfig.epochs):
    actor_model.train()
    critic_model.train()
    epoch_start_time = time.time()

    for step, batch in enumerate(train_loader):
        prompts = batch["prompt"]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=TrainConfig.max_seq_len, padding_side="left"
        ).to(accelerator.device)
        prompt_length = enc.input_ids.shape[1]

        with torch.no_grad():
            model_for_gen = accelerator.unwrap_model(actor_model)
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=TrainConfig.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )

        responses_text = [
            tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]
        rewards = calculate_rewards(
            prompts, responses_text, reward_model, reward_tokenizer,
            accelerator.device, TrainConfig.reasoning
        )

        full_mask = (gen_out != tokenizer.pad_token_id).long()

        with accelerator.accumulate(actor_model, critic_model):
            # Critic: estimate value for each sequence
            values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)
            last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
            values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]
            advantages = rewards - values.detach()

            # Actor: compute log probs over response tokens
            with accelerator.autocast():
                res = actor_model(input_ids=gen_out, attention_mask=full_mask)
                logits = res.logits

            labels = gen_out[:, 1:].clone()
            logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            seq_len = gen_out.size(1) - 1
            resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
            final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))
            actor_logp = (logp_tokens * final_mask).sum(dim=1)

            # Old actor & ref: compute log probs (no grad)
            with torch.no_grad():
                old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits
                old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
                old_logp = (old_logp_tokens * final_mask).sum(dim=1)

                ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits
                ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
                ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)

            # PPO clipped surrogate loss
            kl = (actor_logp - old_logp).mean()
            kl_ref = (actor_logp - ref_logp).mean()
            ratio = torch.exp(actor_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainConfig.clip_epsilon, 1.0 + TrainConfig.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, rewards)
            loss = policy_loss + TrainConfig.vf_coef * value_loss + TrainConfig.kl_coef * kl_ref

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(actor_model.parameters(), TrainConfig.grad_clip)
                accelerator.clip_grad_norm_(critic_model.parameters(), TrainConfig.grad_clip)
                actor_optimizer.step()
                critic_optimizer.step()
                actor_scheduler.step()
                critic_scheduler.step()
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                # Logging
                if optimizer_step % TrainConfig.log_interval == 0:
                    response_ids = gen_out[:, prompt_length:]
                    is_eos = (response_ids == tokenizer.eos_token_id)
                    eos_indices = torch.argmax(is_eos.int(), dim=1)
                    has_eos = is_eos.any(dim=1)
                    lengths = torch.where(has_eos, eos_indices + 1,
                                          torch.tensor(response_ids.shape[1], device=gen_out.device))
                    avg_len = lengths.float().mean().item()

                    actor_lr = actor_scheduler.get_last_lr()[0]
                    critic_lr = critic_scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start_time
                    steps_per_sec = TrainConfig.log_interval / elapsed if elapsed > 0 else 0
                    remaining = (total_steps - optimizer_step) / steps_per_sec if steps_per_sec > 0 else 0

                    accelerator.print(
                        f"Epoch [{epoch+1}/{TrainConfig.epochs}] | Step [{optimizer_step}/{total_steps}] | "
                        f"Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | "
                        f"Reward: {rewards.mean().item():.4f} | KL: {kl.item():.4f} | KL_ref: {kl_ref.item():.4f} | "
                        f"Avg Resp Len: {avg_len:.1f} | Actor LR: {actor_lr:.2e} | "
                        f"Speed: {steps_per_sec:.2f} steps/s | ETA: {remaining:.0f}s"
                    )

                    if swanlab:
                        swanlab.log({
                            "policy_loss": policy_loss.item(),
                            "value_loss": value_loss.item(),
                            "reward": rewards.mean().item(),
                            "kl": kl.item(),
                            "kl_ref": kl_ref.item(),
                            "avg_response_len": avg_len,
                            "actor_lr": actor_lr,
                            "critic_lr": critic_lr,
                        })

                    epoch_start_time = time.time()

                # Sync old actor with current actor
                if optimizer_step % TrainConfig.update_old_actor_freq == 0:
                    raw_actor = accelerator.unwrap_model(actor_model)
                    old_actor_model.load_state_dict(raw_actor.state_dict())

                # Save checkpoint
                if optimizer_step % TrainConfig.save_interval == 0:
                    if accelerator.is_main_process:
                        checkpoint_path = os.path.join(TrainConfig.save_dir, f"{TrainConfig.save_weight}_step{optimizer_step}")
                        accelerator.save_state(checkpoint_path)
                        accelerator.print(f"Saved checkpoint to {checkpoint_path}")

        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp
        del old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss

# Save final model
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(actor_model)
    unwrapped_model.save_pretrained(TrainConfig.model_path)
    tokenizer.save_pretrained(TrainConfig.model_path)
    accelerator.print(f"Training completed! Final model saved to {TrainConfig.model_path}")

if swanlab:
    swanlab.finish()
