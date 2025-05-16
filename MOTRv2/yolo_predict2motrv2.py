

'''
    把yolo检测结果转换为motrv2格式

'''

import os
import os.path as osp
import json
from tqdm import tqdm
import sys


def yolo_predict_to_json(predict_path, json_file):

    os.makedirs(osp.dirname(json_file), exist_ok=True)

    json_dict = {}
    sub_paths = os.listdir(predict_path)
    sub_paths.sort()

    vid_count = 0

    for subpath in sub_paths:

        if 'vis' in subpath:
            continue

        vid_count += 1
        predict_lists = os.listdir(osp.join(predict_path, subpath))
        predict_lists.sort()

        for predict_file in tqdm(predict_lists, desc=subpath):
            json_key = osp.join(subpath, predict_file)
            predict_file = osp.join(predict_path, subpath, predict_file)
            with open(predict_file, 'r') as f:
                lines = f.readlines()
            json_value = lines
            json_dict[json_key] = json_value
    
    with open(json_file, 'w') as f:
        json.dump(json_dict, f)
        print(f'Dump json to file {json_file}. Vid count: {vid_count}')

if __name__ == "__main__":
    predict_path = sys.argv[1]
    json_file = sys.argv[2]
    yolo_predict_to_json(predict_path, json_file)