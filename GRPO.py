import re
import torch
from datasets import load_dataset, Dataset
from sympy import false
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType




# Load model directly

#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
#model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


SYSTEM_PROMPT = """
按照如下格式生成：
<think>
...
</think>
<answer>
...
</answer>

question：2+3=？
<think>
Let me calculate it，2+3=5
</think>

<answer>
5
</answer>

"""

def process_data(data):
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['answer_only']
    }) 
    return data


def extract_answer(text):
    # 情况1：有<answer>标签
    if "<answer>" in text and "</answer>" in text:
        try:
            # 提取标签内内容
            content = text.split("<answer>")[1].split("</answer>")[0]
            # 去除多余空白和换行
            content = re.sub(r'\s+', ' ', content).strip()
            # 尝试提取数字
            numbers = re.findall(r'-?\d+\.?\d*', content)
            if numbers:
                return numbers[0]
            return content
        except:
            pass

    # 情况2：无标签但包含关键短语
    for phrase in ["所以", "因此", "答案是", "answer is"]:
        if phrase in text:
            # 在关键短语后查找数字
            match = re.search(fr"{phrase}.*?(-?\d+\.?\d*)", text)
            if match:
                return match.group(1)

    # 情况3：直接查找最终答案（通常是最后一个数字）
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]  # 取最后一个数字作为答案

    # 情况4：全部失败则返回空
    return ""

def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

# 生成答案是否正确的奖励
def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0][-1]['content']}", f"\n答案:\n{answer[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answer)]


# 格式奖励
def hard_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]
# 格式奖励
def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]
# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [mark_num(response) for response in responses]


if __name__ == '__main__':
    #model_name = "E:/LLM/ModelsUse/Qwen2.5-0.5B-Instruct"
    #model_name = "lora_sft_output/merged_model"
    model_name = "sft_output/final_model"
    model = AutoModelForCausalLM.from_pretrained(model_name)



    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = load_dataset('E:/LLM/DataUse/gsm8k_chinese')
    data = process_data(ds['train'])

    output_dir="output"

    training_args = GRPOConfig(
        output_dir=output_dir,

        learning_rate=1e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=10,
        bf16=False,
        fp16=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=128,
        num_train_epochs=0.3,
        save_steps=100,
        max_grad_norm=1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard"
    )

    trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        mark_reward,
        soft_format_reward,
        hard_format_reward,
        correctness_reward
        ],
    args=training_args,
    train_dataset=data,


)
    trainer.train()
    trainer.save_model(output_dir)
