import sys
import pandas as pd
import json

sys.path.append("/mnt/petrelfs/zhoubaichuan/projects/synthbench")

from tqdm import tqdm

from lm_evaluate.models import GPT


DATA_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/GPT-4o.csv"
SAVE_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/GPT-4o-with-summary-new.csv"

model = GPT(model_version="gpt-4o", api_url="https://api.claudeshop.top/v1/chat/completions")

model.prepare_model()

df = pd.read_csv(DATA_PATH, sep=",")

columns = df.columns

ORIGINAL_TEXT_COLUMN = "原文"
LANGUAGE = "语言"

CHINESE_PROMPT = "这是一段文字: {text}. 请帮我概括这段文字的大概内容。"
CHINESE_POST_PROMPT = """
请按照json的格式输出内容，参考这个格式：
{
    "content": string // 总结内容
}
"""
ENGLISH_PROMPT = "This is a passage: {passage}. Please help me summarize the main content of this passage."
ENGLISH_POST_PROMPT = """
Please output your response in a json object:
{
    'content': string // Your generated summary.
}
"""


data_list = []

for i in tqdm(range(0, len(df), 3)):
    data = df.loc[i]
    original_text = data[ORIGINAL_TEXT_COLUMN]
    
    if data[LANGUAGE] == "中文":
        prompt = CHINESE_PROMPT.format(text=original_text) + CHINESE_POST_PROMPT
    else:
        prompt = ENGLISH_PROMPT.format(passage=original_text) + ENGLISH_POST_PROMPT
    
    summary = model.generate(visuals=None, contexts=prompt, response_format={"type": "json_object"}) 
    
    try:
        summary = json.loads(summary)["content"]
    
    except Exception as e:
        print(e)
    df.loc[i:i+2, "summary"] = summary

df.to_csv(SAVE_PATH, index=False)

