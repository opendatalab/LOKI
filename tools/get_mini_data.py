import json
import random

data_list = json.load(open('/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/synthbench_video_multi_choice.json', 'r'))

mini_data_list = random.sample(data_list, 10)


    
json.dump(mini_data_list, open('/mnt/petrelfs/zhoubaichuan/projects/synthbench/data/synthbench_video_multi_choice_mini.json', 'w'), indent=4)