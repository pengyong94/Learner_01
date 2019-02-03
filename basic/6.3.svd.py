#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
    a[a < 0] = 0
    a[a > 255] = 255
    # a = a.clip(0, 255)
    return np.rint(a).astype('uint8')


def restore2(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K+1):
        for i in range(m):
            a[i] += sigma[k] * u[i][k] * v[k]
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')


if __name__ == "__main__":
    A = Image.open("6.son.png", 'r')  # 通过PIL库中的Image包打开图片文件
    output_path = r'.\Pic'            # 输出路径
    if not os.path.exists(output_path): # 如果输出路径不存在，则创建输出路径。
        os.mkdir(output_path)
    a = np.array(A)   # 将图片信息构建为数组信息array。
    K = 50
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])    # 对图像信息矩阵a做svd分解，并取0,1,2序列矩阵作为分解矩阵
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])
    plt.figure(figsize=(10,10), facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']   # 设置输出图像中写入中文
    mpl.rcParams['axes.unicode_minus'] = False
    for k in range(1, K+1):
        print(k)
        R = restore1(sigma_r, u_r, v_r, k)
        G = restore1(sigma_g, u_g, v_g, k)
        B = restore1(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), 2)     # 将
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
        if k <= 12:
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')
            plt.title(u'奇异值个数：%d' % k)
    plt.suptitle(u'SVD与图像分解', fontsize=18)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.9)
    plt.show()
