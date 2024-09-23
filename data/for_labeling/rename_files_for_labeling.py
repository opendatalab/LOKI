import json
import os
import shutil
import uuid

MODALITIES = ["audio", "image", "video", "3D"]
PARENT_DIR = ["/mnt/hwfile/opendatalab/bigdata_rs/datasets/synthbench-audio/data/AudioDataset_wav", '/mnt/hwfile/opendatalab/bigdata_rs/datasets/FakeImage_dataset', '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/VideoDataset_zbc', '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/3Dpair_dataset']
FILES_TARGET_DIR = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/boyue_for_all_labeling"

os.makedirs(FILES_TARGET_DIR, exist_ok=True)

for i, modality in enumerate(MODALITIES):
    if i < 3:
        continue
    print(modality)
    with open(os.path.join('/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/', modality, 'tf_file_path_all.txt'), 'r') as file:
        os.makedirs(os.path.join(FILES_TARGET_DIR, modality), exist_ok=True)
        file_mappings = {}
        with open(os.path.join(FILES_TARGET_DIR, modality, "file_path_list.txt"), "w") as file_path_list:
            for line in file:
                line = line.strip()
                file_extension = os.path.splitext(line)[1]
                
                mapped_file_name = str(uuid.uuid4()) + file_extension
                
                if modality != "video":
                    shutil.copy2(os.path.join(PARENT_DIR[i], line), os.path.join(FILES_TARGET_DIR, modality, mapped_file_name))
                    file_mappings[mapped_file_name] = os.path.join(PARENT_DIR[i], line)
                else:
                    if os.path.exists(os.path.join(PARENT_DIR[i], line)):
                        shutil.copy2(os.path.join(PARENT_DIR[i], line), os.path.join(FILES_TARGET_DIR, modality, mapped_file_name))
                        file_mappings[mapped_file_name] = os.path.join(PARENT_DIR[i], line)
                    else:
                        parent_dir = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/VideoDataset_zbc_split"
                        shutil.copy2(os.path.join(parent_dir, line), os.path.join(FILES_TARGET_DIR, modality, mapped_file_name))
                        file_mappings[mapped_file_name] = os.path.join(parent_dir, line)
                
                assert os.path.exists(file_mappings[mapped_file_name])
                
                file_path_list.write(mapped_file_name + "\n")
                
        os.makedirs(os.path.join(FILES_TARGET_DIR, modality), exist_ok=True)
        json.dump(file_mappings, open(os.path.join(FILES_TARGET_DIR, modality, "file_mappings.json"), "w"), indent=4)
        
        