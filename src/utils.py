import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, GenerationConfig
import torch


def get_model(model_name="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1"):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config = quantization_config
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def prompt2token(tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"]
    return input_ids, attention_mask

def get_trainingArguments(output_dir="results", epochs=3, batch_size=1):
    return TrainingArguments(
        output_dir=output_dir, 
        overwrite_output_dir=True, 
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, 
        warmup_steps=500, 
        weight_decay=0.01, 
        logging_dir="./logs", 
        logging_steps=20, 
        save_steps=500, 
        evaluation_strategy="no", 
        eval_steps=500,
        save_total_limit=5, 
        fp16=True 
    )

def get_generationConfig(tokenizer, temperature=1, top_p=0.5, top_k=3, no_repeat_ngram_size=3):
    return GenerationConfig(
        do_sample=True,
        temperature=temperature,
        num_beams=1,
        top_p=top_p,
        top_k=top_k,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,  # 終了記号を設定
        eos_token_id=tokenizer.eos_token_id,  # 終止符を添加
    )

def generate_training_data(data_point, tokenizer=None, CUTOFF_LEN=2048):
    data_point = data_point["conversation"]
    prompt = f"""\
        [INST] <<SYS>>
       あなたは裁判官です。
        <</SYS>>

        あなたは裁判官です。以下の要旨の判決を出す際の理由を考えてください。
        要旨：{data_point[0]["content"]}
        [/INST]"""

    len_user_prompt_tokens = (
        len(
            tokenizer(
                prompt,
                truncation=True,
                max_length=256 + 1,
                padding="max_length",
            )["input_ids"]
        ) - 1
    )
    # transform input prompt into tokens
    full_tokens = tokenizer(
        prompt + " 理由：" + data_point[1]["content"] + "</s>",
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

def generate_evaluate_data(tokenizer, model, data_point, generation_config, max_len=1024):
    data_point = data_point["conversation"]
    prompt = f"""\
        [INST] <<SYS>>
       あなたは裁判官です。
        <</SYS>>

        あなたは裁判官です。以下の要旨の判決を出す際の理由を考えてください。
        要旨：{data_point[0]["content"]}
        [/INST]"""
    input_ids, attention_mask =  prompt2token(tokenizer, prompt)
    # 結果を生成
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
        attention_mask=attention_mask
    )
    # 生成されたレスポンスをデコードしてプリントアウト
    for s in generation_output.sequences:
        output = tokenizer.decode(s, skip_special_tokens=True)
        output = output.split("[/INST]")[1].replace("</s>", "").replace("<s>", "").replace("Assistant:", "").replace("Assistant", "").strip()
    return output

def get_pretrained_lora(save_dir="results"):
    ckpts = []
    for ckpt in os.listdir(save_dir):
        if ckpt.startswith("checkpoint-"):
            ckpts.append(ckpt)

    ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[-1]))
    print("利用可能なすべてのチェックポイント:")
    print(" id: checkpoint name")
    for (i, ckpt) in enumerate(ckpts):
        print(f"{i:>3}: {ckpt}")

    id_of_ckpt_to_use = -1  # 1番最後のモデル
    return os.path.join(save_dir, ckpts[id_of_ckpt_to_use])

