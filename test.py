#!/user/bin/python
#coding:UTF-8
from skimage import io,transform
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import glob
import keras as ke
import cv2


def cv_imread(file_path):             #主要是用于cv2读取中文目录文件的问题
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def draw(flag, path):
    color = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    def draw_rectangle(n):
        d = 800
        w = 554
        cv2.rectangle(im, (433+(n-1)*w, 600), (433 + n*w, 600 + d), color[n], 4)

    new_path = "result/bad_image/" + path.split("/")[-1]
    im = cv2.imread(path)
    for i in flag:
        draw_rectangle(i)
    plt.imshow(im)
    plt.savefig(new_path)
    plt.show()


# path = "test/20171.jpg"

# draw([1, 2], path)

a = ['1', '2', '3']

b =list(map(lambda x: int(x), a))
print(b)