import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import os
import warnings

import utils
warnings.simplefilter("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ========== パラメータとか ==========
model_save_dir = "results_3"
test_data_path = "./dataset/test.json"
output_path = "./content"
max_len = 1024 
temperature = 0.1 
top_p = 0.5 
top_k = 3 
# ===================================

model_path = utils.get_pretrained_lora(save_dir=model_save_dir)
tokenizer, model = utils.get_model()
model = PeftModel.from_pretrained(model, model_path)


os.makedirs(output_path, exist_ok=True) 
with open(test_data_path, "r", encoding="utf-8") as f:
    test_datas = json.load(f)

generation_config = utils.get_generationConfig(tokenizer, 
                                               temperature=temperature, top_p=top_p, top_k=top_k)

# 推論を実行して保存する
with open(f"{output_path}/results.txt", "w", encoding="utf-8") as f:
    for (i, test_data) in enumerate(test_datas):
        predict = utils.generate_evaluate_data(tokenizer, model, test_data, generation_config, max_len=max_len)
        f.write(f"{i+1}. " + predict + "\n")
        print(f"{i+1}. " + predict)