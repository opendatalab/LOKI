import json
import random

META_DATA_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/video/video_detection.json"

data_list = json.load(open(META_DATA_PATH, "r"))


mc_list = []
tf_list = []

for data in data_list:
    if data["metric"] == 'open-ended':
        tf_list.append(data)
    elif data["metric"] == "multi-choice" and "fake_from_real" in data["question_type"]:
        mc_list.append(data)
    
print(len(mc_list))
        
sampled_mc_list = random.sample(mc_list, 100)
sampled_tf_list = random.sample(tf_list, 100)

json.dump(sampled_mc_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/video/video-multi-choice.json", "w"), indent=4)
json.dump(sampled_tf_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/video/video-tf.json", "w"), indent=4)
