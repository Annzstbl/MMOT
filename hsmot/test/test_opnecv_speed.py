import cv2
import numpy as np
import time



# 测试一个8通道1200*1200的数据，通过cv2.subtract和cv2.multiply的时间


def func(img, mean, std):
    stdinv = 1 / np.float64(std.reshape(1, -1))
    mean = np.float64(mean.reshape(1, -1))
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


if __name__ == '__main__':
    img = np.random.rand(1200, 1200, 8)
    mean = np.array([0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373])
    std = np.array([0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951])
    mean = mean * 255
    std = std * 255
    img = img.astype(np.float32)


    start_time = time.time()
    counts = 100
    for i in range(counts):
        img_ = func(img, mean, std)
    end_time = time.time()
    print(f'subtract and multiply time: {(end_time - start_time) / counts}')