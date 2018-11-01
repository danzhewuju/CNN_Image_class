#!/user/bin/python
from skimage import io,transform
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import glob


def run(data):
    flower_dict = {0: "不合格", 1: "合格"}
    w = 100
    h = 100
    c = 3
    result = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./train_dir/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./train_dir/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)

        # 打印出预测矩阵
        # print(classification_result)
        # print(classification_result.shape)
        # 打印出预测矩阵每一行最大值的索引
        # print(tf.argmax(classification_result, 1).eval())
        # 根据索引通过字典对应图片的分类
        output = []
        output = tf.argmax(classification_result, 1).eval()
        for i in range(len(output)):
            result.append(flower_dict[output[i]])
            # print("%d产品检状况:" % (i+1) + flower_dict[output[i]])
    return result


def read_paths(path):
    path = "datasets/test"
    paths = glob.glob(path + "/*.jpg")
    return paths


def test(path):
    paths = read_paths(path)
    f = open("result/test_result.txt", 'w', encoding='UTF-8')
    for p_tem in paths:
        data = io.imread(p_tem)
        data = data[290:, 433:2095]
        ims = np.hsplit(data, 3)
        data_tm = []
        for p in ims:
            data_tm.append(transform.resize(p, (100, 100, 3), mode='constant'))
        data_tm = np.array(data_tm)
        result = run(data_tm)
        if result.count("合格") == 3:
            result = "合格"
        else:
            tem = []
            for index in range(result.__len__()):
                if result[index] == "不合格":
                    tem.append(str(index+1))
            result = "不合格:"
            tem_s = ",".join(tem)
            result += tem_s + "\n"
        f.write(result)
        print("图片%s的测试结果为:%s" % (p_tem, result))
    f.close()
    print("Testing Finished")


if __name__ == '__main__':
    path = "datasets/test"
    test(path)










