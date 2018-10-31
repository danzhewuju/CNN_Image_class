from skimage import io,transform
import matplotlib.pyplot as plt
import os
import glob
import os
import tensorflow as tf
import numpy as np


def read_image_all(path):
    paths = []                                          #获取文件夹下所有的图片路径
    for root, dir, files in os.walk(path):
        for f in files:
            pt = os.path.join(root, f)
            paths.append(pt)
    imgs = []

    for index, p in enumerate(paths):
        img_tem = io.imread(p)
        imgs.append(img_tem)
        print("正在读取%s照片" % p)
    return imgs                       #文件下的所有数据信息


def split_image(re_path, save_path, label):
    imgs = read_image_all(re_path)
    count = 1
    for img in imgs:
        # if label == "good":
        #     img = img[350:1793, 400:2041]
        # else:
        #     img = img[605:, 470:2111]
        img = img[290:, 433:2095]
        img_tem = np.hsplit(img, 3)
        for img_t in img_tem:
            path = save_path + "/" + str(count) + "_{0}.jpg".format(label)
            io.imsave(path, img_t)
            print("第%d条图片已经被保存" % count)
            count += 1
    print("图片处理完成！")
    return True


# def split_image(re_path):
#     imgs = read_image_all(re_path)
#     count = 1

split_image("image_data/good", "image_data/good_split", "good")
split_image("image_data/bad", "image_data/bad_split", "bad")