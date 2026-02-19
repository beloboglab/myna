from transformers import Qwen3Config, Qwen3ForCausalLM
def load_model():
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
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    return model

if __name__ == "__main__":
    load_model()