'''
    生成一个txt包含所有图片的绝对路径
'''
import os



train_path = 'path/to/datset/train/npy'
test_path = 'path/to/datset/test/npy'

test_txt = '$ROOT_PATH$/ultralytics/hsmot/cfg_data/train_ch8.txt'
test_txt = '$ROOT_PATH$/ultralytics/hsmot/cfg_data/test_ch8.txt'


def gen_txt(root_path, save_txt):
    with open(save_txt, 'w') as f:
        for path in os.listdir(root_path):
            print(path)
            files = os.listdir(os.path.join(root_path, path))
            files.sort()
            for file in files:
                f.write(f"{os.path.join(root_path, path, file)}/n")


gen_txt(train_path, train_txt)
gen_txt(test_path, test_txt)
