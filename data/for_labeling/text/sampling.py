import json
import random

META_DATA_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_human/text_detection_for_human.json"

data_list = json.load(open(META_DATA_PATH, "r", encoding="utf-16"))


mc_list = []
tf_list = []

for data in data_list:
    if data["question_type"] == 'single':
        tf_list.append(data)
    elif data["question_type"] == "multi":
        mc_list.append(data)
    
print(len(mc_list))
print(len(tf_list))
        
# sampled_mc_list = random.sample(mc_list, 288)
# sampled_tf_list = random.sample(tf_list, 576)

json.dump(mc_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/text/text-multi-choice.json", "w", encoding="utf-16"), indent=4, ensure_ascii=False)
json.dump(tf_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/text/text-tf.json", "w", encoding="utf-16"), indent=4, ensure_ascii=False)
