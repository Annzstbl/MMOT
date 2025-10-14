'''
    生成一个txt包含所有图片的绝对路径
'''
import os


pwd = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.join(pwd, '..'))
DATA_PATH = os.path.abspath(os.path.join(pwd, '..', 'data'))

train_path = os.path.join(DATA_PATH, 'train/npy')
test_path = os.path.join(DATA_PATH, 'test/npy')

train_txt = os.path.join(ROOT_PATH, 'TrackByDetection/ultralytics/mmot/cfg_data/train_8ch.txt')
test_txt = os.path.join(ROOT_PATH, 'TrackByDetection/ultralytics/mmot/cfg_data/test_8ch.txt')


def gen_txt(root_path, save_txt):
    with open(save_txt, 'w') as f:
        for path in os.listdir(root_path):
            print(path)
            files = os.listdir(os.path.join(root_path, path))
            files.sort()
            for file in files:
                f.write(f"{os.path.join(root_path, path, file)}\n")


gen_txt(train_path, train_txt)
gen_txt(test_path, test_txt)
