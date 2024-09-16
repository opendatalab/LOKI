import json
import os
import shutil
import uuid

MODALITIES = ["audio", "image", "video", "3D"]
PARENT_DIR = ["/mnt/hwfile/opendatalab/bigdata_rs/datasets/synthbench-audio/data/AudioDataset_wav", '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset', '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/VideoDataset_zbc', '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/3Dpair_dataset']
FILES_TARGET_DIR = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/for_labeling"


for i, modality in enumerate(MODALITIES):
    with open(os.path.join('/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/', modality, 'tf_file_path.txt'), 'r') as file:
        os.makedirs(os.path.join(FILES_TARGET_DIR, modality), exist_ok=True)
        file_mappings = []
        with open(os.path.join(FILES_TARGET_DIR, modality, "file_path_list.txt"), "w") as file_path_list:
            for line in file:
                mapping = {}
                line = line.strip()
                file_extension = os.path.splitext(line)[1]
                
                mapped_file_name = str(uuid.uuid4()) + file_extension
                
                shutil.copy2(os.path.join(PARENT_DIR[i], line), os.path.join(FILES_TARGET_DIR, modality, mapped_file_name))
                file_mappings.append({line: mapped_file_name})
                
                file_path_list.write(mapped_file_name + "\n")
            
            
        
        json.dump(file_mappings, open(os.path.join(FILES_TARGET_DIR, modality, "file_mappings.json"), "w"), indent=4)