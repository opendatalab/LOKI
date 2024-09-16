import json
from pathlib import Path
import uuid

tf_sampled_data_path = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/text/text-tf.json"

tf_sampled_data = json.load(open(tf_sampled_data_path, "r", encoding="utf-16"))
    

mc_tf_list = []
for i, tf in enumerate(tf_sampled_data):
    label_sample = {
        "prompt": "请判断这段文本内容是真实的还是由AI生成的。"
    }
    
    label_sample["conversation"] = [
        {
            "message_id": str(uuid.uuid4()),
            "content": tf["text_list"][0],
            "message_type": "send",
            "user_id": "",
            "parent_id": None
        },
        {
            "message_id": str(uuid.uuid4()),
            "content": "",
            "message_type": "receive",
            "user_id": "",
            "parent_id": None
        }
    ]
    label_sample["custom"] = {"id": tf["id"]}
    mc_tf_list.append(label_sample)

with open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/text/text_specification.jsonl", "w") as file:
    for data in mc_tf_list:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        