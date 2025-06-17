import re
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

SYSTEM_PROMPT = """Generate the response in the following format:
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
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置pad_token（重要！）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_sft_data(example):

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example['question']}  # 修正：应该是question而不是answer
    ]
    # 构建完整的训练文本（提示 + 模型应有的响应）
    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    # 创建模型应有的响应（包含思考过程和答案）
    response = f"<think>\n让我们一步步解决这个问题。\n{example['question']}\n</think>\n<answer>\n{example['answer']}\n</answer>"

    # 完整的训练文本
    full_text = text + response + tokenizer.eos_token

    # 编码文本
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None
    )

    # 返回必要的字段
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }


if __name__ == '__main__':
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 配置LoRA参数
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,  # 训练模式
        r=4,  # LoRA rank
        lora_alpha=32,  # LoRA缩放参数
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none",
        modules_to_save=None,  #
    )

    # 使用LoRA包装模型
    model = get_peft_model(model, lora_config)

    # 启用训练模式
    model.train()

    # 确保LoRA参数需要梯度
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    # 打印可训练参数信息
    model.print_trainable_parameters()

    # 加载数据集
    ds = load_dataset('E:/LLM/DataUse/gsm8k_chinese')

    # ==================== LoRA 训练阶段 ====================
    print("Starting LoRA SFT training...")

    # 准备SFT数据
    sft_data = ds['train'].map(
        format_sft_data,
        remove_columns=ds['train'].column_names,
        num_proc=1
    )

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # LoRA SFT训练参数
    sft_args = TrainingArguments(
        output_dir="lora_sft_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=20,
        save_steps=500,
        fp16=True,
        optim="adamw_torch",
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        warmup_steps=100,
        weight_decay=0.01,
        save_strategy="steps",
        logging_dir="./logs",
    )

    # LoRA SFT训练器
    sft_trainer = Trainer(
        model=model,
        args=sft_args,
        train_dataset=sft_data,
        data_collator=data_collator
    )

    # 执行LoRA训练
    sft_trainer.train()

    # 保存LoRA适配器
    model.save_pretrained("lora_sft_output/final_lora_adapter")
    tokenizer.save_pretrained("lora_sft_output/final_lora_adapter")

    print("LoRA SFT training completed and adapter saved.")

    # 可选：合并LoRA权重并保存完整模型
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("lora_sft_output/merged_model")
    tokenizer.save_pretrained("lora_sft_output/merged_model")
    print("Merged model saved.")



