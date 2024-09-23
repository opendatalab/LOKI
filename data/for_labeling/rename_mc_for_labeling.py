import json
import os
import shutil
import uuid

MODALITIES = ["video", "audio"]
# PARENT_DIR = ['/mnt/hwfile/opendatalab/bigdata_rs/datasets/FakeImage_dataset', '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/3Dpair_dataset']
PARENT_DIR = ['/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/VideoDataset_zbc', '/mnt/hwfile/opendatalab/bigdata_rs/datasets/synthbench-audio/data/AudioDataset_wav']
FILES_TARGET_DIR = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/boyue_for_labeling"


for i, modality in enumerate(MODALITIES):
    with open(os.path.join('/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/for_labeling/', modality, 'specification.jsonl'), 'r') as file:
        os.makedirs(os.path.join(FILES_TARGET_DIR, modality), exist_ok=True)
        file_mappings = {}
        with open(os.path.join(FILES_TARGET_DIR, modality, "specification.jsonl"), "w") as file_path_list:
            for line in file:
                data = json.loads(line)
                
                media_paths = data["pathList"]
                
                mapped_file_names = []
                for media_path in media_paths:
                    file_extension = os.path.splitext(media_path)[1]
                
                    mapped_file_names.append(str(uuid.uuid4()) + file_extension)
                
                for mapped_file_name, media_path in zip(mapped_file_names, media_paths):
                    if modality != "video":
                        shutil.copy2(os.path.join(PARENT_DIR[i], media_path), os.path.join(FILES_TARGET_DIR, modality, mapped_file_name))
                        file_mappings[mapped_file_name] = os.path.join(PARENT_DIR[i], media_path)
                    else:
                        if os.path.exists(os.path.join(PARENT_DIR[i], media_path)):
                            shutil.copy2(os.path.join(PARENT_DIR[i], media_path), os.path.join(FILES_TARGET_DIR, modality, mapped_file_name))
                            file_mappings[mapped_file_name] = os.path.join(PARENT_DIR[i], media_path)
                        else:
                            parent_dir = "/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/VideoDataset_zbc_split"
                            shutil.copy2(os.path.join(parent_dir, media_path), os.path.join(FILES_TARGET_DIR, modality, mapped_file_name))
                            file_mappings[mapped_file_name] = os.path.join(parent_dir, media_path)
                    
                    assert os.path.exists(file_mappings[mapped_file_name])
                    
                file_path_list.write(json.dumps({"name": data["name"], "pathList": mapped_file_names}) + "\n")
            
        json.dump(file_mappings, open(os.path.join(FILES_TARGET_DIR, modality, "mc_file_mappings.json"), "w"), indent=4)