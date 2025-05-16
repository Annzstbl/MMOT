'''
    把所有的video和image重新命名

    耗时, 不要用debug模式
'''


import os
import os.path as osp
import shutil

ROOT_PATH = './data/hsmot'


NPY_PATH = os.path.join(ROOT_PATH, 'npy')
LOG_PATH = os.path.join(ROOT_PATH, 'rename_log.log')
RGB_PATH = os.path.join(ROOT_PATH, 'rgb')
MOT_PATH = os.path.join(ROOT_PATH, 'mot')

# NPY_PATH = '/data3/PublicDataset/Custom/HSMOT/npy'
# LOG_PATH = '/data3/PublicDataset/Custom/HSMOT/log.log'
# RGB_PATH = '/data3/PublicDataset/Custom/HSMOT/rgb'
# MOT_PATH = '/data3/PublicDataset/Custom/HSMOT/mot'


def rename_video(root_path):
    video_list = os.listdir(root_path)
    video_list.sort()
    # 视频命名 data23-20-80m-2146-1616-129-wu_00_394_463 , 根据data23重新命名为data23-1 data23-2
    video_data_dict = {}
    with open(LOG_PATH, 'a') as f:
        for video in video_list:
            video_name = video.split('-')[0].split('_')[0]
            if video_name not in video_data_dict:
                video_data_dict[video_name] = 1
            else:
                video_data_dict[video_name] += 1
            new_video_name = video_name + '-' + str(video_data_dict[video_name])
            f.write(video + ' -> ' + new_video_name + '\n')
            print('rename:', video, '->', new_video_name)
            shutil.move(osp.join(root_path, video), osp.join(root_path, new_video_name))

def rename_mot_label(root_path):
    label_list = os.listdir(root_path)
    label_list.sort()
    # 视频命名 data23-20-80m-2146-1616-129-wu_00_394_463 , 根据data23重新命名为data23-1 data23-2
    video_data_dict = {}
    with open(LOG_PATH, 'a') as f:
        for video in label_list:
            ext = os.path.splitext(video)[-1]
            video_name = video.split('-')[0].split('_')[0]
            if video_name not in video_data_dict:
                video_data_dict[video_name] = 1
            else:
                video_data_dict[video_name] += 1
            new_video_name = video_name + '-' + str(video_data_dict[video_name]) + ext
            f.write(video + ' -> ' + new_video_name + '\n')
            print('rename:', video, '->', new_video_name)
            shutil.move(osp.join(root_path, video), osp.join(root_path, new_video_name))

def rename_file(video_path):
    file_list = os.listdir(video_path)
    file_list.sort()
    ext = os.path.splitext(file_list[0])[-1]
    # 文件命名 xxx.npy -> 000001.npy
    with open(LOG_PATH, 'a') as f:
        for i, file in enumerate(file_list):
            new_file = str(i+1).zfill(6) + ext
            f.write(file + ' -> ' + new_file + '\n')
            print('rename:', file, '->', new_file)
            shutil.move(osp.join(video_path, file), osp.join(video_path, new_file))
    

if __name__ == '__main__':

    rename_mot_label(MOT_PATH)

    rename_video(NPY_PATH)
    for video in os.listdir(NPY_PATH):
        video_path = osp.join(NPY_PATH, video)
        rename_file(video_path)

    rename_video(RGB_PATH)
    for video in os.listdir(RGB_PATH):
        video_path = osp.join(RGB_PATH, video)
        rename_file(video_path)