import json
from pathlib import Path

tf_sampled_data_path = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/audio/audio-tf-all.json"
mc_sampled_data_path = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/audio/audio-multi-choice-all.json"

image_dir = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/synthbench-audio/data/"

tf_file_path_txt = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/audio/tf_file_path_all.txt"

tf_sampled_data = json.load(open(tf_sampled_data_path, "r"))
mc_sampled_data = json.load(open(mc_sampled_data_path, "r"))


with open(tf_file_path_txt, "w") as f:
    for tf in tf_sampled_data:
        image_path = tf["audio_path"]
        relative_path = image_path.replace('AudioDataset/', "")
        f.write(relative_path + "\n")
    

mc_label_list = []
for i, mc in enumerate(mc_sampled_data):
    label_sample = {}
    
    image_paths = mc["audio_path"]
    
    label_sample["name"] = str(i)
    label_sample["pathList"] = [image_path.replace('AudioDataset/', "") for image_path in image_paths]
    mc_label_list.append(label_sample)

with open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/audio/audio_specification_all.jsonl", "w") as file:
    for data in mc_label_list:
        file.write(json.dumps(data) + "\n")