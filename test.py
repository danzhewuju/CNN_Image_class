#!/user/bin/python
from skimage import io,transform
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from CNN_test import run
import os
import glob


path = "image_data/bad/20170911190956_75011772147501177214O11121702447708_Pr000_e[]三对pin焊点偏.jpg"
data = io.imread(path)
data = data[290:, 433:2095]
ims = np.hsplit(data, 3)
data_tm = []
for p in ims:
    data_tm.append(transform.resize(p, (100, 100, 3)))
data_tm = np.array(data_tm)
run(data_tm)

