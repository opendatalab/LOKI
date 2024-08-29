import sys
import json
import pandas as pd

sys.path.append("/mnt/petrelfs/zhoubaichuan/projects/synthbench")

from tqdm import tqdm

from lm_evaluate.models import GPT


DATA_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/GPT-4o-with-summary-new.csv"
SAVE_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/GPT-4o-generated-new.csv"

model = GPT(model_version="gpt-4o", api_url="https://api.claudeshop.top/v1/chat/completions")

model.prepare_model()

df = pd.read_csv(DATA_PATH)

columns = df.columns

ORIGINAL_TEXT_COLUMN = "原文"
LANGUAGE = "语言"

GENERATED = "生成"

DIFFICULTY_LEVEL = "难度"
WORD_COUNT = "字数"
TEXT_TYPE = "类型"

CHINESE_PROMPT = "你是一个{role}. 这是一段{text_type}类型的文章的总结。请帮我依据这段总结，生成{word_count}字的{text_type}文章。总结：{summary}"
CHINESE_POST_PROMPT = """
请使用json格式输出你的回答,参考这个格式:
{
    'content': string // 根据总结生成的内容
}
"""
ENGLISH_PROMPT = "You are a {role}. This is a summary of a {text_type} passage. Based on the summary, please help me generate a {word_count}-word {text_type} passage. Summary: {summary}."
ENGLISH_POST_PROMPT = """
Please output your response in a json object:
{
    'content': string // Your generated content based on the summary.
}
"""
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
        prompt = CHINESE_PROMPT.format(role=role, word_count=word_count, summary=summary, text_type=text_type) + CHINESE_POST_PROMPT
    else:
        role, word_count, summary, text_type = ENGLISH_ROLES[data[TEXT_TYPE]], data[WORD_COUNT], data['summary'], ENGLISH_TEXT_TYPE[data[TEXT_TYPE]]
        
        if role == "ancient literary scholar":
            prompt = ENGLISH_PROMPT.format(role=role, word_count=word_count, summary=summary, text_type=text_type) + ENGLISH_POST_PROMPT
            prompt += "Please output your response in modern English."
    
    passage = model.generate(visuals=None, contexts=prompt, response_format={"type": "json_object"})
    
    try:
        passage = json.loads(passage)["content"]
    except Exception as e:
        print(e)

    
    df.loc[i, GENERATED] = passage
    df.loc[i, DIFFICULTY_LEVEL] = "易"
    
    sample["model"] = "gpt-4o"
    sample["generated_content"] = passage
    sample["original_content"] = data[ORIGINAL_TEXT_COLUMN]
    sample["summary"] = data["summary"]
    sample["difficulty_level"] = "易"
    sample["prompt"] = prompt
    sample["content_type"] = text_type
    
    data_list.append(sample)

df.to_csv(SAVE_PATH, index=False)

with open('/mnt/petrelfs/zhoubaichuan/projects/synthbench/text_data/gpt_generated_new_prompt.json', 'w',  encoding="utf-16") as f:
    json.dump(data_list, f, indent=4, ensure_ascii=False)
    
    