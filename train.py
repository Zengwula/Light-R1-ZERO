import re
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
import trl
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = """按照如下格式生成：
思考：
...
答案：
<answer>
...
</answer>
"""

# 提前加载tokenizer
model_name = "./Qwen2.5-0.5B-Instruct"#模型存储的位置
tokenizer = AutoTokenizer.from_pretrained(model_name)


def format_sft_data(example):
    """准备SFT训练数据格式"""
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example['question_zh-cn']}
    ]
    # 构建完整的训练文本（提示 + 模型应有的响应）
    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    # 创建模型应有的响应（包含思考过程和答案）
    response = f"思考：让我们一步步解决这个问题。首先，{example['question_zh-cn'].split('。')[0]}...\n" \
               f"答案：\n<answer>\n{example['answer_only']}\n</answer>"

    # 编码文本为input_ids（返回列表而不是张量）
    input_ids = tokenizer.encode(
        text + response,
        truncation=True,
        max_length=512,
        return_tensors=None  # 返回列表而不是张量
    )

    # 返回input_ids
    return {"input_ids": input_ids}


def process_data(data):
    """准备GRPO训练数据"""
    return data.map(lambda x: {
        'prompt': tokenizer.apply_chat_template([
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ], tokenize=False, add_generation_prompt=True),
        'answer': x['answer_only']
    })


def extract_answer(text):
    """从模型输出中提取答案"""
    if "<answer>" in text and "</answer>" in text:
        try:
            return text.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            pass
    return ""


def format_reward(completions,  ** kwargs):
    """格式奖励函数"""
    rewards = []
    for comp in completions:
        text = comp[0]['content']
        reward = 0.0

        # 基本格式元素检测
        if "思考：" in text: reward += 0.2
        if "答案：" in text: reward += 0.2
        if "<answer>" in text: reward += 0.2
        if "</answer>" in text: reward += 0.2

        # 尝试提取答案内容
        extracted = extract_answer(text)
        if extracted:
            reward += 0.2  # 有提取内容
            if extracted.isdigit():  # 是数字
                reward += 0.2

        rewards.append(min(reward, 1.0))  # 上限1.0
    return rewards


def correctness_reward(prompts, completions,  ** kwargs):
    """正确性奖励函数"""
    answers = kwargs["answer"]
    rewards = []

    for i, (comp, ans) in enumerate(zip(completions, answers)):
        text = comp[0]['content']
        extracted = extract_answer(text)
        ans = str(ans)
        reward = 0.0

        # 完全正确的情况
        if extracted == ans:
            reward = 2.0
        # 部分正确的情况
        elif extracted and ans in extracted:
            reward = 1.0
        # 数字接近的情况
        elif extracted.isdigit() and ans.isdigit():
            diff = abs(float(extracted) - float(ans))
            reward = max(0, 1.0 - diff / 10.0)  # 根据误差给分
        # 有提取内容但错误
        elif extracted:
            reward = 0.3

        rewards.append(reward)
    return rewards


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 加载数据集
    ds = load_dataset('./gsm8k_chinese')#数据集存储的位置

    # ==================== SFT训练阶段 ====================
    print("Starting SFT training...")

    # 准备SFT数据
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=50,
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

    # ==================== GRPO训练阶段 ====================
    print("Starting GRPO training...")

    # 重新加载SFT训练后的模型
    model = AutoModelForCausalLM.from_pretrained("sft_output/final_model")

    # 准备GRPO数据
    grpo_data = process_data(ds['train']).select(range(500))  # 使用部分数据加速训练

    # GRPO训练参数
    training_args = GRPOConfig(
        output_dir="grpo_output",
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=10,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=2,  # 响应数量
        max_prompt_length=384,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=500,
        max_grad_norm=0.2,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard"
    )

    # GRPO训练器
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs_and_kwargs=[
            (format_reward, {}),
            (correctness_reward, {"answer": "answer"})
        ],
        args=training_args,
        train_dataset=grpo_data,
    )

    # 执行GRPO训练
    trainer.train()
    trainer.save_model("grpo_output/final_model")
    print("GRPO training completed.")