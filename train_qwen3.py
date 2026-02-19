#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3模型训练脚本
使用transformers库和mock数据集进行训练
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import json
from typing import Dict, List


class MockDataset(Dataset):
    """Mock数据集生成器"""
    
    def __init__(self, tokenizer, num_samples=1000, max_length=512):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.data = self._generate_mock_data()
    
    def _generate_mock_data(self) -> List[str]:
        """生成mock训练数据"""
        mock_texts = []
        
        # 生成一些模拟的中文对话和文本数据
        templates = [
            "用户：{query}\n助手：{response}",
            "问题：{query}\n答案：{response}",
            "输入：{query}\n输出：{response}",
        ]
        
        mock_queries = [
            "什么是人工智能？",
            "如何学习Python？",
            "解释一下机器学习的基本概念",
            "深度学习有哪些应用？",
            "自然语言处理的主要任务是什么？",
            "如何训练一个语言模型？",
            "Transformer架构的核心是什么？",
            "什么是注意力机制？",
        ]
        
        mock_responses = [
            "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "学习Python可以从基础语法开始，然后逐步学习数据结构、面向对象编程和常用库。",
            "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式，而无需明确编程。",
            "深度学习在图像识别、自然语言处理、语音识别、推荐系统等领域有广泛应用。",
            "自然语言处理的主要任务包括文本分类、情感分析、机器翻译、问答系统等。",
            "训练语言模型需要大量文本数据、合适的模型架构、训练策略和计算资源。",
            "Transformer架构的核心是自注意力机制，能够并行处理序列数据并捕获长距离依赖。",
            "注意力机制允许模型在处理序列时关注不同位置的信息，提高对相关信息的理解。",
        ]
        
        for i in range(self.num_samples):
            template = templates[i % len(templates)]
            query = mock_queries[i % len(mock_queries)]
            response = mock_responses[i % len(mock_responses)]
            text = template.format(query=query, response=response)
            mock_texts.append(text)
        
        return mock_texts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def prepare_dataset(tokenizer, num_samples=1000, max_length=512):
    """准备训练数据集"""
    print(f"生成 {num_samples} 个mock样本...")
    dataset = MockDataset(tokenizer, num_samples=num_samples, max_length=max_length)
    
    # 转换为HuggingFace Dataset格式
    data_dict = {
        'input_ids': [],
        'attention_mask': []
    }
    
    for item in dataset:
        data_dict['input_ids'].append(item['input_ids'].tolist())
        data_dict['attention_mask'].append(item['attention_mask'].tolist())
    
    hf_dataset = HFDataset.from_dict(data_dict)
    return hf_dataset


def main():
    # 配置参数
    model_name = "Qwen/Qwen2.5-0.5B"  # 使用Qwen2.5作为基础模型（Qwen3可能还未发布）
    # 如果Qwen3已发布，可以改为 "Qwen/Qwen3-xxx"
    
    output_dir = "./qwen3_checkpoints"
    num_train_samples = 1000
    max_length = 512
    batch_size = 4
    num_epochs = 3
    learning_rate = 5e-5
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载tokenizer和模型
    print(f"加载模型和tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # 准备数据集
    train_dataset = prepare_dataset(
        tokenizer,
        num_samples=num_train_samples,
        max_length=max_length
    )
    
    print(f"数据集大小: {len(train_dataset)}")
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型不使用MLM
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="no",
        save_strategy="steps",
        fp16=device == "cuda",  # 使用混合精度训练
        dataloader_pin_memory=True,
        report_to="none",  # 不使用wandb等
        remove_unused_columns=False,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print(f"保存模型到 {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("训练完成！")


if __name__ == "__main__":
    main()
