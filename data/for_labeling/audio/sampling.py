import json
import random

META_DATA_PATH = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/synthbench-audio/data/AudioDataset_wav/synthbench_audio.json"

data_list = json.load(open(META_DATA_PATH, "r"))


mc_list = []
tf_list = []

option_map = {"A": 0, "B": 1}

for data in data_list:
    if data["question_type"] == 'single':
        tf_list.append(data)
    elif data["question_type"] == "multi":
        mc_list.append(data)
    
print(len(mc_list))
        
sampled_mc_list = random.sample(mc_list, 200)
sampled_tf_list = random.sample(tf_list, 200)

json.dump(sampled_mc_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/audio/audio-multi-choice.json", "w"), indent=4)
json.dump(sampled_tf_list, open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/audio/audio-tf.json", "w"), indent=4)