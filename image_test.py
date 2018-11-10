#!/user/bin/python
from skimage import io,transform
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import glob
import cv2

w = 100
h = 100
c = 1


def cv_imread(file_path):             #主要是用于cv2读取中文目录文件的问题
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def draw(flag, path):
    color = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    def draw_rectangle(n):
        d = 800
        w = 554
        cv2.rectangle(im, (433+(n-1)*w, 600), (433 + n*w, 600 + d), color[n], 4)

    new_path = "result/bad_image/" + path.split("\\")[-1]
    im = cv_imread(path)
    for i in flag:
        draw_rectangle(i)
    plt.imshow(im)
    plt.savefig(new_path)
    plt.show()


def run(data):
    path = "datasets/photos"
    cate = [x for x in os.listdir(path) if os.path.isdir(path + "/" + x)]
    flower_dict = {}
    if cate[0] == "bad_split":
        flower_dict = {0: "不合格", 1: "合格"}
    else:
        flower_dict = {0: "合格", 1: "不合格"}
    result = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./train_dir/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./train_dir/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)

        output = []
        output = tf.argmax(classification_result, 1).eval()
        for i in range(len(output)):
            result.append(flower_dict[output[i]])
            # print("%d产品检状况:" % (i+1) + flower_dict[output[i]])
    return result


def read_paths(path):
    path = "./test"
    # paths = glob.glob(path + "/*.jpg")
    paths = glob.glob(os.path.join(path, "*.jpg"))
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
            data_tm.append(transform.resize(p, (w, h, c), mode='constant'))
        data_tm = np.array(data_tm)
        result = run(data_tm)
        if result.count("合格") == 3:
            result = "合格\n"
        else:
            tem = []
            for index in range(result.__len__()):
                if result[index] == "不合格":
                    tem.append(str(index+1))
            result = "不合格:"
            #在此圈出不合格的商品
            bad_index = list(map(lambda x: int(x), tem))
            draw(bad_index, p_tem)    #保存统计信息
            tem_s = ",".join(tem)
            result += tem_s + "\n"
        f.write(result)
        print("图片%s的测试结果为:%s" % (p_tem, result))

    f.close()
    print("Testing Finished")


if __name__ == '__main__':
    path = "./test"
    test(path)










