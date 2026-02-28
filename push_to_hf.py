#!/usr/bin/env python3
"""
认证（任选其一）：
  - 推荐：hf auth login  
  - 或设置环境变量 HF_TOKEN
  - 或直接运行本脚本，按提示输入 token
"""
import os
from pathlib import Path

MODEL_DIR = "final/myna_25M_pretrain"
# 目标 HF 仓库 ID，例如 "your_username/myna_25M"
REPO_ID = os.environ.get("HF_REPO_ID", "beloboglab/myna_25M_pretrain")


def main():
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        raise ImportError("请先安装: pip install huggingface_hub")

    # 未设置 HF_TOKEN 时会提示输入 token，无需 huggingface-cli
    login()

    api = HfApi()
    # 若仓库不存在则先创建（避免 Repository Not Found）
    print(f"检查/创建仓库: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print(f"正在上传: {MODEL_DIR} -> https://huggingface.co/{REPO_ID}")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"完成: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
