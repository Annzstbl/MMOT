'''
    把mot转成yolo
'''


import os
import cv2
import numpy as np 
from tqdm import tqdm

def mot_to_yolo(mot_dir, img_dir, output_dir):
    # 创建输出目录，如果不存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历 MOT 文件
    for mot_file in os.listdir(mot_dir):
        if mot_file.endswith('.txt'):
            vid_name = os.path.splitext(mot_file)[0]  # 视频名
            mot_path = os.path.join(mot_dir, mot_file)

            with open(mot_path, 'r') as f:
                lines = f.readlines()

            # 存储每帧的标注
            mot_label_dict = {}
            for line in lines:
                frame, obj_id, x0, y0, x1, y1, x2, y2, x3, y3, _, cls, _ = map(float, line.split(','))
                if int(frame) not in mot_label_dict:
                    mot_label_dict[int(frame)] = []
                mot_label_dict[int(frame)].append((cls, x0, y0, x1, y1, x2, y2, x3, y3))

            # 图像列表和初始图像读取
            img0_file = os.path.join(img_dir, vid_name, os.listdir(os.path.join(img_dir, vid_name))[0])
            img0 = cv2.imread(img0_file)
            img_height, img_width = img0.shape[:2]  # 获取图像的高和宽

            yoloobb_output_path = os.path.join(output_dir, vid_name)
            os.makedirs(yoloobb_output_path, exist_ok=True)

            # 处理每一帧的标注
            for frame_idx in tqdm(sorted(mot_label_dict.keys()), desc=vid_name):
                yolo_file_path = os.path.join(yoloobb_output_path, f"{frame_idx:06d}.txt")
                with open(yolo_file_path, 'w') as yolo_file:
                    for obj in mot_label_dict[frame_idx]:
                        cls, x0, y0, x1, y1, x2, y2, x3, y3 = obj
                        # 假设 x0, x1, x2, x3 是你的坐标
                        x_coordinates = np.array([x0, x1, x2, x3,])
                        normalized_x_coordinates = x_coordinates / img_width
                        y_coordinates = np.array([y0, y1, y2, y3])
                        normalized_y_coordinates = y_coordinates / img_height

                        x0, x1, x2, x3 = normalized_x_coordinates
                        y0, y1, y2, y3 = normalized_y_coordinates

                        # yolo不允许超范围的坐标
                        if all(0 <= val <= 1 for val in normalized_x_coordinates[:4]) and all(0 <= val <= 1 for val in normalized_y_coordinates[:4]) and 0<= cls <= 7:
                            # 转换为 OBB 格式
                            yolo_file.write(f"{int(cls)} {x0:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f}\n")
                        else:
                            pass
                            # print(f'delete lines: {normalized_x_coordinates}, {normalized_y_coordinates}, {cls}')

def check_yolo(root_path):
    '''检查yolo标注是否有负数和大于1的值'''
    for sub_path in os.listdir(root_path):
        for txt in tqdm(os.listdir(os.path.join(root_path, sub_path)), desc=sub_path):
            txt_path = os.path.join(root_path, sub_path, txt)
            # 存储有效的行
            valid_lines = []
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # 分割行并转换为浮点数
                    values = list(map(float, line.split()))
                    
                    # 检查前八个元素
                    if len(values) == 9 and all(0 <= val <= 1 for val in values[:9]):
                        valid_lines.append(line)  # 保留合法行
                    
            # 写回有效的行
            with open(txt_path, 'w') as f:
                f.writelines(valid_lines)



if __name__ == "__main__":
    root_path = '/data/users/litianhao/data/HSMOT'

    mot_test_dir = os.path.join(root_path, 'test', 'mot')
    img_test_dir = os.path.join(root_path, 'test', 'rgb')
    output_test_dir = os.path.join(root_path, 'test', 'yolo_det_labels')

    mot_to_yolo(mot_test_dir, img_test_dir, output_test_dir)
    
    mot_train_dir = os.path.join(root_path, 'train', 'mot')
    img_train_dir = os.path.join(root_path, 'train', 'rgb')
    output_train_dir = os.path.join(root_path, 'train', 'yolo_det_labels')

    mot_to_yolo(mot_train_dir, img_train_dir, output_train_dir)

    # check_yolo("/data3/PublicDataset/Custom/HSMOT/train/labels")
