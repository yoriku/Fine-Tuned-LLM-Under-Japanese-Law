import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
from threading import Thread
import os
import utils
import warnings
warnings.simplefilter("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ========== パラメータとか ==========
model_save_dir = "results_3"
CUTOFF_LEN = 2048
epochs = 3
batch_size = 1
# ===================================

def generate_training_data(data_point, CUTOFF_LEN=CUTOFF_LEN):
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

tokenizer, model = utils.get_model()

# LoRAアダプタの設定を定義
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM,
)
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_peft_model(model, lora_config)
model = model.to(device)

# JSONファイルからデータセットを読み込みプロンプトを作成する
data = load_dataset('json', data_files="dataset/output.json")
train_data = data['train'].shuffle().map(generate_training_data)
val_data = None

# Fine Tuningの実行
training_args = utils.get_trainingArguments(output_dir=model_save_dir, epochs=epochs, batch_size=batch_size)
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args, 
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False

# PyTorch バージョン 2.0 以降且つ Windows 以外のシステムを使用している場合、モデルをコンパイル？
if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

trainer.train()

model.save_pretrained("backup")