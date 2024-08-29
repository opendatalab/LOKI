import json
import os
import math

import pandas as pd
import numpy as np

META_DATA_PATH = '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/VideoDataset_zbc/video_inf.csv'
DST_DATA_PATH = '/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/synthbench_video_meta.json'
VIDEO_DATA_PATH = '/mnt/hwfile/opendatalab/bigdata_rs/datasets/Synthbench/VideoDataset_zbc/'


df = pd.read_csv(META_DATA_PATH, encoding='ISO-8859-1')

data_list = []

FAKE_COLUMN = "Fake Video Name"
REAL_COLUMN = "Real Video Name"
model_column = "Method"


for i in range(len(df)):
    fake_sample = {}
    real_sample = {}
    
    model = df.loc[i, model_column]
    fake_video_name = df.loc[i, FAKE_COLUMN]
    real_video_name = df.loc[i, REAL_COLUMN]
    
    if not isinstance(real_video_name, str):
        continue
    
    fake_video_path = os.path.join(VIDEO_DATA_PATH, 'fake_video', model, fake_video_name)
    real_video_path = os.path.join(VIDEO_DATA_PATH, 'real_video', model, real_video_name)
    
    if not os.path.exists(real_video_path) or not os.path.exists(fake_video_path):
        print(real_video_path, fake_video_path)
        
    
    fake_sample['video_path'] = fake_video_path
    real_sample['video_path'] = real_video_path
    
    fake_sample['model'] = model
    real_sample['model'] = model
    
    fake_sample['is_fake'] = True
    real_sample['is_fake'] = False
    
    data_list.append(fake_sample)
    data_list.append(real_sample)
    

json.dump(data_list, open(DST_DATA_PATH, 'w'), indent=4)