#!/user/bin/python
from skimage import io,transform
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from CNN_test import run
import os
import glob


path_good = "datasets/test/20170011180011_75011772147501177214O11121702447575_Pr000_e[]中pin、右pin焊点偏.jpg"
path_bad = "datasets/test/20180002210002_264_P45_75011772607501177260O02101802994100_e5f1649120744c57a62c134203ddc848.jpg"
data = io.imread(path_bad)
data = data[290:, 433:2095, :]
ims = np.hsplit(data, 3)
run(ims)

