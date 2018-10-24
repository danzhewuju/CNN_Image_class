from skimage import io,transform
import matplotlib.pyplot as plt
import os
import glob
import os
import tensorflow as tf
import numpy as np


data = io.imread("image_data/bad/20170004140033_75011771757501177175O04081702171711_Pr000_e[]三个pin焊点偏，右pin焊黑.jpg")
data_a = data[605:, 470:2111]
imgs = np.hsplit(data_a, 3)
for img in imgs:
    io.imshow(img)
    plt.show()