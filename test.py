#!/user/bin/python
from skimage import io,transform
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import glob
import keras as ke

path = "image_data/good/20180002220017_469_P45_75011772607501177260O02101802994203_d4770d605eba4295b4f33d5229b99979.jpg"
data = io.imread(path)
data = transform.resize(data, (100, 100, 3), mode='constant')
io.imshow(data)
plt.show()
print(data.shape)
