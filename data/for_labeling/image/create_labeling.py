import json
from pathlib import Path

tf_sampled_data_path = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/image/image-tf.json"
mc_sampled_data_path = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/image/image-multi-choice.json"

image_dir = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset"

tf_file_path_txt = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/image/tf_file_path.txt"

tf_sampled_data = json.load(open(tf_sampled_data_path, "r"))
mc_sampled_data = json.load(open(mc_sampled_data_path, "r"))


with open(tf_file_path_txt, "w") as f:
    for tf in tf_sampled_data:
        image_path = tf["image_path"]
        relative_path = Path(image_path).relative_to(Path(image_dir))
        f.write(str(relative_path) + "\n")
    

mc_label_list = []
for i, mc in enumerate(mc_sampled_data):
    label_sample = {}
    
    image_paths = mc["image_path"]
    
    label_sample["name"] = str(i)
    label_sample["pathList"] = [str(Path(image_path).relative_to(Path(image_dir))) for image_path in image_paths]
    mc_label_list.append(label_sample)

with open("/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/image/image_specification.jsonl", "w") as file:
    for data in mc_label_list:
        file.write(json.dumps(data) + "\n")