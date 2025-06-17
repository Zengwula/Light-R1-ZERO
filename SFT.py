import re
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
import trl
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT =  """Generate the response in the following format:
<thinking>
...
</thinking>
<answer>
...
</answer>

Example:
User Question: What is 2+3?
<thinking>
Let me calculate: 2 plus 3 equals 5.
</thinking>
<answer>
5
</answer>
"""

# 提前加载tokenizer
model_name = "E:/LLM/ModelsUse/Qwen2.5-0.5B-Instruct"
#model_name = "E:/LLM/ModelsUse/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def format_sft_data(example):
    """准备SFT训练数据格式"""
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example['answer']}
    ]
    # 构建完整的训练文本（提示 + 模型应有的响应）
    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    # 创建模型应有的响应（包含思考过程和答案）
    response = f"思考：让我们一步步解决这个问题。首先，{example['question'].split('。')[0]}...\n" \
               f"答案：\n<answer>\n{example['answer']}\n</answer>"

    # 编码文本为input_ids（返回列表而不是张量）
    input_ids = tokenizer.encode(
        text + response,
        truncation=True,
        max_length=512,
        return_tensors=None  # 返回列表而不是张量
    )

    # 返回input_ids
    return {"input_ids": input_ids}





if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 加载数据集
    ds = load_dataset('E:/LLM/DataUse/gsm8k_chinese')

    # ==================== SFT训练阶段 ====================
    print("Starting SFT training...")

    # 准备SFT数据 - 只保留input_ids字段
    sft_data = ds['train'].map(
        format_sft_data,
        remove_columns=ds['train'].column_names  # 移除所有原始列
    )

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 使用因果语言建模（Causal LM）
    )

    # SFT训练参数
    sft_args = TrainingArguments(
        output_dir="sft_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=2,
        logging_steps=20,
        save_steps=500,
        fp16=True,
        optim="adamw_torch",
        report_to="tensorboard",
        remove_unused_columns=False
    )

    # SFT训练器
    sft_trainer = Trainer(
        model=model,
        args=sft_args,
        train_dataset=sft_data,
        data_collator=data_collator  # 使用数据整理器创建标签
    )

    # 执行SFT训练
    sft_trainer.train()

    # 保存SFT模型
    model.save_pretrained("sft_output/final_model")
    tokenizer.save_pretrained("sft_output/final_model")
    print("SFT training completed and model saved.")

