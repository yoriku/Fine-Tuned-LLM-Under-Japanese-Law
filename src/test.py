from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread
import utils

tokenizer, model = utils.get_model()

chat = [
    { "role": "user", "content": "まどか☆マギカでは誰が一番かわいい?" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# add_generation_prompt=Trueにすることで、<|start_header_id|>assistant<|end_header_id|>\n\nまでがプロンプトテンプレートに含まれる。らしい..
# https://qiita.com/lighlighlighlighlight/items/48ed6532c481d78a139d

input_ids, attention_mask = utils.prompt2token(tokenizer, prompt)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# モデルの生成結果をリアルタイムに逐次取得

def generate():
    model.generate(
        input_ids,
        max_new_tokens=512,
        streamer=streamer,
        attention_mask=attention_mask
    )
thread = Thread(target=generate) # 以下，すれっとで実行
thread.start()
for text in streamer:
    print(text, end="", flush=True)
thread.join()   # 待機するコードらしい