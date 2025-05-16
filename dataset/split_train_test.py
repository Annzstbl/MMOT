'''
    把数据集划分为训练集和测试集
'''

import os
import shutil
import os.path as osp

TEST_LIST = '''
data23-1
data24-1
data27-1
data28-1
data28-2
data28-3
data28-4
data28-5
data28-6
data30-2
data30-3
data30-4
data30-5
data30-9
data30-10
data31-1
data33-1
data34-1
data34-2
data34-3
data36-4
data36-5
data36-10
data36-11
data36-12
data36-13
data37-1
data37-2
data37-10
data37-11
data37-12
data39-1
data39-2
data39-3
data39-6
data40-3
data40-4
data40-5
data41-1
data42-1
data42-2
data42-3
data46-11
data46-12
data47-1
data47-2
data47-3
data47-4
data48-1
data49-2
'''

TRAIN_LIST = '''
data23-2
data23-3
data25-1
data25-2
data27-2
data27-3
data30-1
data30-6
data30-7
data30-8
data35-1
data36-1
data36-2
data36-3
data36-6
data36-7
data36-8
data36-9
data37-3
data37-4
data37-5
data37-6
data37-7
data37-8
data37-9
data39-4
data39-5
data40-1
data40-2
data42-4
data42-5
data43-1
data43-2
data43-3
data43-4
data43-5
data43-6
data43-7
data43-8
data44-1
data44-2
data44-3
data46-1
data46-2
data46-3
data46-4
data46-5
data46-6
data46-7
data46-8
data46-9
data46-10
data48-2
data48-3
data49-1
data50-1
data50-2
data50-3
data50-4
data50-5
data51-1
data51-2
data51-3
data51-4
data51-5
data51-6
data52-1
data52-2
data52-3
data52-4
data52-5
data52-6
data52-7
data52-8
data52-9
'''


TEST_LIST = TEST_LIST.strip().split('\n')
TRAIN_LIST = TRAIN_LIST.strip().split('\n')
ALL_LIST = TEST_LIST + TRAIN_LIST

def check_vid_name(root_path):

    rst = True

    npy_path = os.path.join(root_path, 'npy')
    rgb_path = os.path.join(root_path, 'rgb')
    mot_path = os.path.join(root_path, 'mot')
    
    npy_set = set()
    rgb_set = set()
    mot_set = set()


    for npy_file in os.listdir(npy_path):
        vid_name = osp.splitext(npy_file)[0]
        npy_set.add(vid_name)
    
    for rgb_file in os.listdir(rgb_path):
        vid_name = osp.splitext(rgb_file)[0]
        rgb_set.add(vid_name)

    for mot_file in os.listdir(mot_path):
        vid_name = osp.splitext(mot_file)[0]
        mot_set.add(vid_name)
    
    #比对ALL_LIST 和 npy_set, rgb_set, mot_set
    if len(ALL_LIST) != len(npy_set) or len(ALL_LIST) != len(rgb_set) or len(ALL_LIST) != len(mot_set):
        print('len(ALL_LIST) != len(npy_set) or len(ALL_LIST) != len(rgb_set) or len(ALL_LIST) != len(mot_set)')
        print('len(ALL_LIST):', len(ALL_LIST))
        print('len(npy_set):', len(npy_set))
        print('len(rgb_set):', len(rgb_set))
        print('len(mot_set):', len(mot_set))
        rst =  False

    for vid_name in ALL_LIST:
        if vid_name not in npy_set:
            print(f'{vid_name} not in npy_set')
            rst = False
        if vid_name not in rgb_set:
            print(f'{vid_name} not in rgb_set')
            rst =  False
        if vid_name not in mot_set:
            print(f'{vid_name} not in mot_set')
            rst =  False

    return rst


def ln_train_test(root_path):
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(osp.join(train_path, 'npy'), exist_ok=True)
    os.makedirs(osp.join(test_path, 'npy'), exist_ok=True)
    os.makedirs(osp.join(train_path, 'rgb'), exist_ok=True)
    os.makedirs(osp.join(test_path, 'rgb'), exist_ok=True)
    os.makedirs(osp.join(train_path, 'mot'), exist_ok=True)
    os.makedirs(osp.join(test_path, 'mot'), exist_ok=True)
    
    npy_path = os.path.join(root_path, 'npy')
    rgb_path = os.path.join(root_path, 'rgb')
    mot_path = os.path.join(root_path, 'mot')


    for npy_file in os.listdir(npy_path):

        dst_file = osp.join(test_path, 'npy', npy_file) if npy_file in TEST_LIST else osp.join(train_path, 'npy', npy_file)
        if osp.exists(dst_file):
            # 删除
            os.remove(dst_file)
        # 创建软链接
        print(f'symlink: {osp.join(npy_path, npy_file)} -> {dst_file}')
        os.symlink(osp.join(npy_path, npy_file), dst_file)

    for rgb_file in os.listdir(rgb_path):
            
        dst_file = osp.join(test_path, 'rgb', rgb_file) if rgb_file in TEST_LIST else osp.join(train_path, 'rgb', rgb_file)
        if osp.exists(dst_file):
            # 删除
            os.remove(dst_file)
        # 创建软链接
        print(f'symlink: {osp.join(rgb_path, rgb_file)} -> {dst_file}')
        os.symlink(osp.join(rgb_path, rgb_file), dst_file)

    for mot_file in os.listdir(mot_path):
                
        dst_file = osp.join(test_path, 'mot', mot_file) if osp.splitext(mot_file)[0] in TEST_LIST else osp.join(train_path, 'mot', mot_file)
        if osp.exists(dst_file):
            # 删除
            os.remove(dst_file)
        # 创建软链接
        print(f'symlink: {osp.join(mot_path, mot_file)} -> {dst_file}')
        os.symlink(osp.join(mot_path, mot_file), dst_file)

if __name__ == '__main__':
    root_path = './data/HSMOT'
    root_path = os.path.abspath(root_path)

    if not check_vid_name(root_path=root_path):
        print('check_vid_name failed')
        exit(1)
    else :
        print('check_vid_name passed')

    ln_train_test(root_path=root_path)