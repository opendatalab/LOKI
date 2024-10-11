import json
import random

META_DATA_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/video/video_detection_0929.json"

data_list = json.load(open(META_DATA_PATH, "r"))


mc_list = []
tf_list = []

for data in data_list:
    if data["metric"] == 'open-ended':
        if 'lumiere' in data['video_path'].lower():
            tf_list.append(data)
    elif data["metric"] == "multi-choice" and "fake_from_real" in data["question_type"]:
        mc_list.append(data)

tf_list = tf_list[0::2]
print(len(mc_list))
print(len(tf_list))
        
# sampled_mc_list = random.sample(mc_list, 26)
# sampled_tf_list = random.sample(tf_list, 69)

# json.dump(mc_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/video/video-multi-choice-lumiere.json", "w"), indent=4)
json.dump(tf_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/video/video-tf-lumiere.json", "w"), indent=4)
