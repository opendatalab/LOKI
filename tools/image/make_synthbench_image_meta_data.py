import json
import os
import pandas as pd

from collections import Counter


ANIMALS_DATA = pd.read_csv("/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/animals_inf.csv", encoding='ISO-8859-1')
MEDICINE_DATA = pd.read_csv("/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/medicine_new/medicine_inf.csv", encoding='ISO-8859-1')
OBJECT_DATA = pd.read_csv("/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/object_inf.csv", encoding='ISO-8859-1')
SATELLITE_DATA = pd.read_csv("/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/satellite.csv", encoding='ISO-8859-1')


META_DATA_PATH = "/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/synthbench_image_meta.json"


data_list = []

# convert animals and objects information

animals_real_image = set()
for i in range(len(ANIMALS_DATA)):
    data = ANIMALS_DATA.loc[i]
    
    real_sample = {}
    fake_sample = {}
    
    image_type = data['type']
    method = data['method']
    
    real_image_name = data['real_image']
    real_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/animals/real', image_type, real_image_name)
    assert os.path.exists(real_image_path), real_image_path
    
    
    fake_image_name = data['fake_image']
    fake_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/animals/fake/type', image_type, fake_image_name)
    assert os.path.exists(fake_image_path), fake_image_path
    
    fake_sample = {"image_path": fake_image_path, "is_fake": True, "image_type": "animals", "image_subtype": image_type, "generation_method": method}
    data_list.append(fake_sample)
    
    if real_image_path not in animals_real_image:
        real_sample = {"image_path": real_image_path, "is_fake": False, "image_type": "animals", "image_subtype": image_type, "generation_method": None}
        data_list.append(real_sample)
        animals_real_image.add(real_image_path)
    

print(len(animals_real_image))
    
object_real_image = set()

for i in range(len(OBJECT_DATA)):
    data = OBJECT_DATA.loc[i]
    
    image_type = data['type']
    image_name = data['real_image']
    real_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/object/real', image_type, image_name)
    assert os.path.exists(real_image_path), real_image_path
    
    fake_image_name = data['fake_image']
    fake_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/object/fake', image_type, fake_image_name)
    assert os.path.exists(fake_image_path), fake_image_path
    
    fake_sample = {"image_path": fake_image_path, "is_fake": True, "image_type": "object", "image_subtype": image_type, "generation_method": method}
    data_list.append(fake_sample)
    
    if real_image_path not in object_real_image:
        real_sample = {"image_path": real_image_path, "is_fake": False, "image_type": "object", "image_subtype": image_type, "generation_method": None}
        data_list.append(real_sample)
        object_real_image.add(real_image_path)


print(len(object_real_image))

print(len(data_list))

# add medicine and satellite

medicine_real_image = set()

for i in range(len(MEDICINE_DATA)):
    data = MEDICINE_DATA.loc[i]
    
    method = data['method']
    
    fake_image_name = data['fake_image']
    fake_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/medicine_new/fake', method, fake_image_name)
    
    assert os.path.exists(fake_image_path), fake_image_path
    
    fake_sample = {"image_path": fake_image_path, "is_fake": True, "image_type": "medicine", "image_subtype": None, "generation_method": method}
    
    data_list.append(fake_sample)
    
    if isinstance(data['real_image'], str):
        real_image_name = data['real_image']
        
        real_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/medicine_new/real', real_image_name)
        assert os.path.exists(real_image_path), real_image_path
        
        if real_image_path not in medicine_real_image:
            
            real_sample = {"image_path": real_image_path, "is_fake": False, "image_type": "medicine", "image_subtype": None, "generation_method": None}
            medicine_real_image.add(real_image_path)
            
            data_list.append(real_sample)
            


satellite_real_image = set()

for i in range(len(SATELLITE_DATA)):
    data = SATELLITE_DATA.loc[i]
    
    method = data['method']
    
    fake_image_name = data['fake_image']
    fake_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/satellite/fake', method, fake_image_name)
    
    assert os.path.exists(fake_image_path), fake_image_path
    
    fake_sample = {"image_path": fake_image_path, "is_fake": True, "image_type": "satellite", "image_subtype": None, "generation_method": method}
    
    data_list.append(fake_sample)
    
    if isinstance(data['real_image'], str):
        real_image_name = data['real_image']
        
        real_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/satellite/real', real_image_name)
        assert os.path.exists(real_image_path), real_image_path
        
        if real_image_path not in satellite_real_image:
            
            real_sample = {"image_path": real_image_path, "is_fake": False, "image_type": "satellite", "image_subtype": None, "generation_method": None}
            satellite_real_image.add(real_image_path)
            
            data_list.append(real_sample)

print(len(data_list))


# prepare person

person_fake_image = list(os.listdir('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/person/fake'))
person_real_image = list(os.listdir('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/person/real'))

for fake in person_fake_image:
    fake_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/person/fake', fake)
    
    method = fake.split("_")[0]
    
    fake_sample = {"image_path": fake_image_path, "is_fake": True, "image_type": "person", "image_subtype": None, "generation_method": method}
    data_list.append(fake_sample)


for real in person_real_image:
    real_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/person/real', real)

    real_sample = {"image_path": real_image_path, "is_fake": False, "image_type": "person", "image_subtype": None, "generation_method": None}
    data_list.append(real_sample)
    

# prepare scene

scene_fake_image = list(os.listdir('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/scene/fake'))
scene_real_image = list(os.listdir('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/scene/real'))

for fake in scene_fake_image:
    fake_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/scene/fake', fake)
    assert os.path.exists(fake_image_path)
    
    method = fake.split("_")[0]
    
    fake_sample = {"image_path": fake_image_path, "is_fake": True, "image_type": "scene", "image_subtype": None, "generation_method": method}
    data_list.append(fake_sample)


for real in scene_real_image:
    real_image_path = os.path.join('/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/FakeImage_dataset/scene/real', real)
    assert os.path.exists(real_image_path)

    real_sample = {"image_path": real_image_path, "is_fake": False, "image_type": "scene", "image_subtype": None, "generation_method": None}
    data_list.append(real_sample)


print(len(data_list))

json.dump(data_list, open(META_DATA_PATH, 'w'), indent=4)


type_counter = Counter()

for sample in data_list:
    if sample['is_fake']:
        im_type = 'fake'
    else:
        im_type = 'real'
    type_counter[f"{sample['image_type']}_{im_type}"] += 1

print(type_counter)

