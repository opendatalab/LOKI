import json
from pathlib import Path
import uuid

tf_sampled_data_path = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/video/video-multi-choice.json"

tf_sampled_data = json.load(open(tf_sampled_data_path, "r"))

mapping = json.load(open("/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/boyue_for_labeling/video/mc_file_mappings.json", "r"))

path_list = list(open("/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/boyue_for_labeling/video/specification.jsonl", "r"))

mc_tf_list = []
for i, tf in enumerate(tf_sampled_data):
    label_sample = {
        "prompt": "下面有两个视频，请判断哪一条视频是由AI生成的。"
    }
    video1 = json.loads(path_list[i])["pathList"][0]
    video2 = json.loads(path_list[i])["pathList"][1]
    
    video_url1 = rf"http://10.140.31.254:80/zhoubaichuan/boyue_for_labeling/video/{video1}"
    video_url2 = rf"http://10.140.31.254:80/zhoubaichuan/boyue_for_labeling/video/{video2}"
    
    label_sample["conversation"] = [
        {
            "message_id": str(uuid.uuid4()),
            "content": f'<html><body><table><tr><td><video width="auto" height="auto" controls><source src="{video_url1}" type="video/mp4">Your browser does not support the video tag.</video></td></tr></table></body></html>',
            "message_type": "send",
            "user_id": "",
            "parent_id": None
        },
        {
            "message_id": str(uuid.uuid4()),
            "content": f'<html><body><table><tr><td><video width="auto" height="auto" controls><source src="{video_url2}" type="video/mp4">Your browser does not support the video tag.</video></td></tr></table></body></html>',
            "message_type": "receive",
            "user_id": "",
            "parent_id": None
        }
    ]
    label_sample["custom"] = {"id": tf["question_type"]}
    mc_tf_list.append(label_sample)

with open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/video/video_mc_specification.jsonl", "w") as file:
    for data in mc_tf_list:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        