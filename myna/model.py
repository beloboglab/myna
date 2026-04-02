from transformers import Qwen3Config, Qwen3ForCausalLM, BitsAndBytesConfig
from typing import Optional, Tuple, Union
 
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import torch
 
 
def _print_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
 
 
def load_model(
    model_path: Optional[str] = None,
    model_format: str = "hf",
    # LoRA 配置
    lora: Optional[Union[str, bool]] = None,  # None/False: 不用；"train"/True: 训练；str路径: 加载适配器
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    # QLoRA（可选）
    use_qlora: bool = False,
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",  # 或 "float16"
    bnb_4bit_quant_type: str = "nf4",          # 或 "fp4"
):
    if model_path is None:
        config = Qwen3Config(
            vocab_size=6400,           # 词表大小，可改成你自己的
            hidden_size=512,             # 隐藏维度
            intermediate_size=1408, #1024,      # FFN 中间层，通常约 4*hidden_size 或 8/3*hidden_size
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,       # 若做 GQA 可小于 num_attention_heads
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_dropout=0.0,
            bias=False,
            tie_word_embeddings=True,
            head_dim=64,
        )
        model = Qwen3ForCausalLM(config)
    else:
        if model_format != "hf":
            raise ValueError(f"Unsupported model format: {model_format}")
 
        quantization_config = None
        if use_qlora:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[bnb_4bit_compute_dtype]
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
            )
 
        if quantization_config is not None:
            model = Qwen3ForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")
        else:
            model = Qwen3ForCausalLM.from_pretrained(model_path)
 
    print(model_path)
 
    # 应用 LoRA/QLoRA
    if lora:
        # 关闭缓存，便于训练
        if hasattr(model, "config"):
            model.config.use_cache = False
        if getattr(model, "gradient_checkpointing_enable", None):
            model.gradient_checkpointing_enable()
 
        if use_qlora:
            # 低比特训练准备
            model = prepare_model_for_kbit_training(model)
 
        # 从已训练的 LoRA 适配器目录加载
        if isinstance(lora, str) and lora not in ("true", "True", "train"):
            base_model = model
            model = PeftModel.from_pretrained(base_model, lora)
            model.enable_input_require_grads()
            print(f"已从 LoRA 适配器加载: {lora}")
        else:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(target_modules),
            )
            model = get_peft_model(model, peft_config)
            model.enable_input_require_grads()
            print("已启用 LoRA 训练封装")
 
    _print_trainable_params(model)
    return model
 
if __name__ == "__main__":
    load_model(model_path="./final/myna_25M")
    load_model(model_path="./final/myna_25M")