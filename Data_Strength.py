import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image
import glob


def reinforce_data(path, save_dir, count):
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    img_path = path
    img = image.load_img(img_path)

    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        # imgplot = plt.imshow(image.array_to_img(batch[0]))
        count += 1
        save_path = os.path.join(save_dir, "{}.png".format(count))
        plt.imsave(save_path, image.array_to_img(batch[0]))
        print("正在加强第{}张照片......".format(count))
        i += 1
        if i % 10 == 0:
            break
    # plt.show()
    return count


def save_image(paths, save_dir, count=0):
    count_t = count
    for p in paths:
        count_t = reinforce_data(p, save_dir, count_t)
    return count


path_good_original = "datasets/photos/bad_split"
path_bad_original = "datasets/photos/good_split"
save_path_bad = "image_data/bad_strength"
save_path_good = "image_data/good_strength"
count = 0

paths_good = [x for x in glob.glob(path_good_original + '/*.jpg')]   #获得所有的照片列表
paths_bad = [p for p in glob.glob(path_bad_original + '/*.jpg')]    #获得所有照片数据

save_image(paths_good, save_path_good, 0)
save_image(paths_bad, save_path_bad, 0)






