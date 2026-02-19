```
# 原代码（加载预训练权重）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

# 改为（随机初始化）
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(
    config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
if device == "cuda":
    model = model.cuda()
else:
    model = model.to(device)
```