import sys
import json
import pandas as pd
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAHD3BSbQZW9Y4I-fvkJN7QIp3eUlgrajA"

sys.path.append("/mnt/petrelfs/zhoubaichuan/projects/synthbench")

from tqdm import tqdm

from lm_evaluate.models import Gemini



DATA_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/GPT-4o-with-summary.csv"
SAVE_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/Gemini-1.5-Pro-generated.csv"

model = Gemini(model_version="gemini-1.5-pro")

model.prepare_model()

df = pd.read_csv(DATA_PATH, encoding="utf-16", sep="\t")

columns = df.columns

ORIGINAL_TEXT_COLUMN = "原文"
LANGUAGE = "语言"

GENERATED = "生成"

DIFFICULTY_LEVEL = "难度"
WORD_COUNT = "字数"
TEXT_TYPE = "类型"

CHINESE_PROMPT = "你是一个{role}. 这是一段{text_type}类型的文章的总结: {summary}。请帮我依据这段总结，生成{word_count}字的{text_type}文章"
ENGLISH_PROMPT = "You are a {role}. This is a summary of a {text_type} passage: {summary}. Based on the summary, please help me generate a {word_count}-word {text_type} passage."
ENGLISH_ROLES = {"古文学": "ancient literary scholar", "哲学": "philosopher", "当代文学": "novelist"}
CHINESE_ROLES = {"古文学": "古文学家", "哲学": "哲学家", "当代文学": "小说家"}

ENGLISH_TEXT_TYPE = {"古文学": "Old English", "哲学": "philosophical", "当代文学": "novel"}

data_list = []

for i in tqdm(range(0, len(df), 3)):
    data = df.loc[i]
    original_text = data[ORIGINAL_TEXT_COLUMN]
    
    text_type = data[TEXT_TYPE]
    
    sample = {}
    
    if data[LANGUAGE] == "中文":
        role, word_count, summary, text_type = CHINESE_ROLES[data[TEXT_TYPE]], data[WORD_COUNT], data['summary'], data[TEXT_TYPE]
        prompt = CHINESE_PROMPT.format(role=role, word_count=word_count, summary=summary, text_type=text_type)
    else:
        role, word_count, summary, text_type = ENGLISH_ROLES[data[TEXT_TYPE]], data[WORD_COUNT], data['summary'], ENGLISH_TEXT_TYPE[data[TEXT_TYPE]]
        prompt = ENGLISH_PROMPT.format(role=role, word_count=word_count, summary=summary, text_type=text_type)
    
    passage = model.generate(visuals=None, contexts=prompt)
    
    df.loc[i, GENERATED] = passage
    df.loc[i, DIFFICULTY_LEVEL] = "易"
    
    sample["model"] = "gemini-1.5-pro"
    sample["generated_content"] = passage
    sample["original_content"] = data[ORIGINAL_TEXT_COLUMN]
    sample["summary"] = data["summary"]
    sample["difficulty_level"] = "easy"
    sample["prompt"] = prompt
    sample["content_type"] = text_type
    
    data_list.append(sample)

df.to_csv(SAVE_PATH, index=False)

with open('/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/gemini_generated.json', 'w',  encoding="utf-16") as f:
    json.dump(data_list, f, indent=4, ensure_ascii=False)