import os
from ultralytics import settings
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import numpy as np
import torch
import sys

if __name__ == "__main__":

    '''
        python hsmot/predict_normal.py /data/users/litianhao/data/hsmot/train/npy /data/users/litianhao/hsmot_code/workdir/yolo11/predict_trainset_yolov11l_8ch_CocoPretrain_CopyFirstConv_imgsize1280_4gpu /data/users/litianhao/hsmot_code/workdir/yolo11/yolov11l_8ch_CocoPretrain_CopyFirstConv_imgsize1280_4gpu/weights/best.pt
    '''


    assert len(sys.argv) == 5, "Please provide the path to the config file."
    TEST_ROOT_PATH = sys.argv[1]
    PREDICT_ROOT_PATH = sys.argv[2]
    WEIGHTS_FILE = sys.argv[3]
    MODE = sys.argv[4] # rgb / npy
    BATCH_SIZE = 4


model = YOLO(WEIGHTS_FILE)

vis_example = 10

for vid in os.listdir(TEST_ROOT_PATH):
    os.makedirs(os.path.join(PREDICT_ROOT_PATH, vid), exist_ok=True)
    
    vis_count = 0# 每个视频序列保存vis_example张结果
    
    for img in tqdm(os.listdir(os.path.join(TEST_ROOT_PATH, vid)), desc=vid):
        img_npy = np.load(os.path.join(TEST_ROOT_PATH, vid, img))
        if MODE == 'rgb':
            img_npy = np.ascontiguousarray(img_npy[..., [1,2,4]])
        result = model(img_npy, imgsz=1280, conf=0.1, iou=0.6)

        if vis_count < vis_example:
            save_img = os.path.join(PREDICT_ROOT_PATH, 'vis', f'{vid}_{os.path.basename(img)}').replace('.npy','.jpg')
            os.makedirs(os.path.dirname(save_img), exist_ok=True)
            result[0].save(save_img)
            vis_count += 1
            
        save_txt = os.path.join(PREDICT_ROOT_PATH, vid, img.replace('.jpg', '.txt').replace('.png', '.txt').replace('.npy', '.txt'))
        with open(save_txt, 'w') as f:
            for xyxyxyxy, cls, score in zip(result[0].obb.xyxyxyxy, result[0].obb.cls, result[0].obb.conf):
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, xyxyxyxy.reshape(-1).cpu().numpy())  # 转换为 NumPy 数组
                cls = int(cls.cpu().numpy())  # 转换为整数
                score = float(score.cpu().numpy())  # 转换为浮点数

                # 将结果写入文件
                f.write(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {cls} {score:.6f}\n")
    # break
