# Copy from MiniMind (https://github.com/jingyaogong/minimind).
# See MiniMind's LICENSE for license terms.

import os
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from safetensors.torch import load_file as safe_load_file

warnings.filterwarnings('ignore')


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_full_hf_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def init_model(args):
    load_from = os.path.expanduser(args.load_from)
    base_model = os.path.expanduser(args.base_model or args.load_from)

    if _is_full_hf_dir(load_from):
        # 标准 HF 目录：config + tokenizer + 权重都在 load_from
        tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            load_from,
            trust_remote_code=True,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
    else:
        # 仅权重的 checkpoint 目录（如 pretrain_step10000）：只有 model.safetensors 等，无 config/tokenizer
        # config 与 tokenizer 从 base_model 取，权重从 load_from 的 model.safetensors 加载
        if not _is_full_hf_dir(base_model):
            raise FileNotFoundError(
                f"checkpoint 目录缺少 config.json，请用 --base_model 指定含 config 与 tokenizer 的 HF 目录，例如：\n"
                f"  --base_model ./final/myna_25M --load_from {load_from}"
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
        ckpt_path = os.path.join(load_from, "model.safetensors")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint 目录下未找到 model.safetensors: {load_from}")
        state_dict = safe_load_file(ckpt_path)
   
        if not any(k.startswith("model.") for k in state_dict):
            state_dict = {"model." + k: v for k, v in state_dict.items()}
        # checkpoint 可能不含 lm_head（与 embed 共享时），用 strict=False 允许缺失
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Loaded] missing keys (e.g. lm_head when tied): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[Loaded] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"[Loaded weights from checkpoint: {ckpt_path}]")

    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="Myna 模型推理与对话")
    parser.add_argument('--load_from', default='./final/myna_25M', type=str, help="模型目录（完整 HF 或仅权重的 checkpoint 目录）")
    parser.add_argument('--base_model', default='./final/myna_25M', type=str, help="当 load_from 为仅权重 checkpoint 时，从此目录加载 config 与 tokenizer")
    parser.add_argument('--max_new_tokens', default=2048, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度（0–1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus 采样阈值（0–1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（偶数，0 表示不携带）")
    parser.add_argument('--show_speed', default=1, type=int, help="是否显示 decode 速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--bf16', action='store_true', default=True, help="使用 bfloat16 推理（默认开启）")
    parser.add_argument('--no_bf16', dest='bf16', action='store_false', help="关闭 bfloat16")
    args = parser.parse_args()

    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0:
            print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        try:
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = (tokenizer.bos_token or "") + prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n')
        else:
            print('\n\n')


if __name__ == "__main__":
    main()
