
# 加载 Gemma 3 1B Instruct，并设置参数
from unsloth import FastModel
import torch
# 设置 GRPO Trainer 和所有配置
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
class grpo_class:
    def __init__(self,config):
        self.model, self.tokenizer = FastModel.from_pretrained(
        model_name = config.model_name,
        max_seq_length = config.max_seq_length, # Choose any for long context!
        load_in_4bit = config.in_4bit,  # 4 bit quantization to reduce memory
        load_in_8bit = config.in_8bit, # [NEW!] A bit more accurate, uses 2x memory
        fast_inference = config.fast_inference,
        max_lora_rank = config.max_lora_rank,
        gpu_memory_utilization = config.gpu_memory_utilization
        )
        #添加了 LoRA 适配器，因此我们只需要更新少量参数！
        self.model = FastModel.get_peft_model(
            self.model,
            finetune_vision_layers     = config.finetune_vision_layers, # Turn off for just text!
            finetune_language_layers   = config.finetune_language_layers,  # Should leave on!
            finetune_attention_modules = config.finetune_attention_modules,  # Attention good for GRPO
            finetune_mlp_modules       = config.finetune_attention_modules,  # SHould leave on always!
            r = config.max_lora_rank,           # Larger = higher accuracy, but might overfit
            lora_alpha = config.lora_alpha,  # Recommended alpha == r at least
            lora_dropout = config.lora_dropout,
            bias = config.bias,
            random_state = config.random_state,
        )
        self.training_args = GRPOConfig(
            learning_rate = 5e-6,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_torch_fused",
            logging_steps = 1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1, # Increase to 4 for smoother training
            num_generations = 4, # Decrease if out of memory
            max_prompt_length = config.max_prompt_length,
            max_completion_length = config.max_seq_length - config.max_prompt_length,
            # num_train_epochs = 1, # Set to 1 for a full training run
            max_steps = 50,
            save_steps = 50,
            max_grad_norm = 0.1,
            report_to = "none", # Can use Weights & Biases
            output_dir = "outputs",
        )

    def train(self,dataset,reward_funcs_list):
        # 创建 GRPO 训练器
        trainer = GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = reward_funcs_list,
            args = self.training_args,
            train_dataset = dataset,
        )
        trainer.train()
    def verify(self):
        # 推理
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "What is the sqrt of 101?"},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = False,
        )
        from transformers import TextStreamer
        _ = self.model.generate(
            **self.tokenizer(text, return_tensors = "pt").to("cuda"),
            max_new_tokens = 64, # Increase for longer outputs!
            # Recommended Gemma-3 settings!
            temperature = 1.0, top_p = 0.95, top_k = 64,
            streamer = TextStreamer(self.tokenizer, skip_prompt = True),
        )
    def save(self):
        # 保存和加载模型
        # [注意] 这只会保存 LoRA 适配器，而不会保存完整模型。要保存为 16 位或 GGUF，请向下滚动！
        self.model.save_pretrained("gemma-3")  # Local saving
        self.tokenizer.save_pretrained("gemma-3")
        # ('gemma-3/tokenizer_config.json',
        #  'gemma-3/special_tokens_map.json',
        #  'gemma-3/tokenizer.model',
        #  'gemma-3/added_tokens.json',
        #  'gemma-3/tokenizer.json')

        # 保存为 VLLM 的 float16
        if False: # Change to True to save finetune!
            model.save_pretrained_merged("gemma-3-finetune", tokenizer)
        if False: # Change to True to upload finetune
            model.push_to_hub_merged(
                "HF_ACCOUNT/gemma-3-finetune", tokenizer,
                token = "hf_..."
            )

        # 要保存到 GGUF / llama.cpp，我们现在为所有型号原生支持它！目前，您可以轻松转换为 Q8_0、F16 或 BF16 精度。4 位 Q4_K_M 稍后推出！
        if False: # Change to True to save to GGUF
            model.save_pretrained_gguf(
                "gemma-3-finetune",
                quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
            )
        if False: # Change to True to upload GGUF
            model.push_to_hub_gguf(
                "gemma-3-finetune",
                quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
                repo_id = "HF_ACCOUNT/gemma-finetune-gguf",
                token = "hf_...",
            )
        # 现在，在 llama.cpp 或基于 UI 的系统（如 Jan 或 Open WebUI）中使用 gemma-3-finetune.gguf 文件或 gemma-3-finetune-Q4_K_M.gguf 文件。
        
#Data Prep  数据准备
# 我们正在使用 OpenAI 著名的 GSM8K 数据集！

dataset = load_dataset("openai/gsm8k", "main", split = "train")

# Dataset({
#     features: ['question', 'answer'],
#     num_rows: 7473
# })

# 让我们看看第一行：
print(dataset[0]["question"])

# 让我们看看第一行：
print(dataset[0]["answer"])

# 提取答案
# 注意到所有答案（如 about）都有一个 ####，因此我们提取它
def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()
extract_hash_answer(dataset[0]["answer"])

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
system_prompt

#绘制数据

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})
dataset[0]
# 创建一个正则表达式格式来匹配推理部分和答案
import re

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)
#验证是否可以提取出来
match_format.search(
    "<start_working_out>Let me think!<end_working_out>"\
    "<SOLUTION>2</SOLUTION>",
)

# Reward
# 创建一个 reward 函数来完全匹配格式 - 如果成功，我们将奖励它 3 分
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores
# 如果失败，我们希望通过计算每个交易品种来奖励至少部分遵循格式的模型：
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
        score += 0.5 if response.count(solution_start)  == 1 else -0.5
        score += 0.5 if response.count(solution_end)    == 1 else -0.5
        scores.append(score)
    return scores
# 最后，我们想要提取生成的答案，并对其进行奖励或惩罚！我们还根据答案与真实答案的接近程度通过比率来奖励它
def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        # 如果看到空格则匹配
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            # 如果答案通过比例接近，我们也会给予奖励！
            # 也就是说，如果答案在某个范围内，就给予奖励！
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                else: score -= 1.0 # Penalize wrong answers
            except:
                score -= 0.5 # Penalize
        scores.append(score)
    return scores

#数字获取测试
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags = re.MULTILINE | re.DOTALL
)
match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>")

#检测数字
def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess       = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores









