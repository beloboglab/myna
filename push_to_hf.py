#!/usr/bin/env python3
"""
将 myna/final/myna_25M 推送到 Hugging Face Hub。
使用前请：
  1. pip install huggingface_hub
  2. huggingface-cli login  或设置环境变量 HF_TOKEN
"""
import os
from pathlib import Path

# 本地模型目录（相对项目根）
MODEL_DIR = "/root/projs/myna/final/myna_25M"
# 目标 HF 仓库 ID，例如 "your_username/myna_25M"
REPO_ID = os.environ.get("HF_REPO_ID", "belobog/myna_25M")


def main():

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("请先安装: pip install huggingface_hub")

    api = HfApi()
    print(f"正在上传: {MODEL_DIR} -> https://huggingface.co/{REPO_ID}")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"完成: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
