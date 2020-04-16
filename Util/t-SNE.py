import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import utils
import numpy as np
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
import math
from sklearn.manifold import TSNE

# 加载源域
path_source = ['S1.npy', 'S2.npy', 'S3.npy', 'S4.npy', 'S5.npy']

for i in [0, 1, 2, 3, 4]:  # 读入数据和标签转换成tensor
    print('-' * 10)
    if i == 0:
        source_data = np.load(path_source[i])

        print('source_data_{}.shape'.format(i), source_data.shape)
        source_labels = np.zeros((source_data.shape[0]))
        print('source_labels_{}.shape'.format(i), source_labels.shape)
    else:
        temp_data = np.load(path_source[i])

        print('source_data_{}.shape'.format(i), temp_data.shape)
        temp_labels = np.zeros((temp_data.shape[0])) + i
        print('source_labels_{}.shape'.format(i), temp_labels.shape)

        source_data = np.vstack((source_data, temp_data))
        source_labels = np.hstack((source_labels, temp_labels))
    print('-' * 10)


# 加载目标域
path_target = ['T1.npy', 'T2.npy', 'T3.npy', 'T4.npy', 'T5.npy']

for i in [0, 1, 2, 3, 4]:  # 读入数据和标签转换成tensor
    print('-' * 10)
    if i == 0:
        target_data = np.load(path_target[i])

        print('target_data_{}.shape'.format(i), target_data.shape)
        target_labels = np.zeros((target_data.shape[0]))
        print('target_labels_{}.shape'.format(i), target_labels.shape)
    else:
        temp_data = np.load(path_target[i])

        print('target_data_{}.shape'.format(i), temp_data.shape)
        temp_labels = np.zeros((temp_data.shape[0])) + i
        print('target_labels_{}.shape'.format(i), temp_labels.shape)

        target_data = np.vstack((target_data, temp_data))
        target_labels = np.hstack((target_labels, temp_labels))
    print('-' * 10)

source_data = np.load(r'A:\我的交大\Transfer Learning\Deep Transfer Learning\条件分布适配\条件分布适配_source.npy')
target_data = np.load(r'A:\我的交大\Transfer Learning\Deep Transfer Learning\条件分布适配\条件分布适配_target.npy')

# 打印读取后source的数据和标签shape
print('-' * 10)
print('source.shape:', source_data.shape)
print('source_labels.shape', source_labels.shape)
print('-' * 10)

# 打印读取后target的数据和标签shape
print('-' * 10)
print('target.shape:', target_data.shape)
print('target_labels.shape', target_labels.shape)
print('-' * 10)

feature = np.vstack((source_data, target_data))
print('feature.shape:', feature.shape)

cValue_source = []
cValue_target = []

for label in source_labels:
    if label == 0:
        cValue_source.append('g')  # green
    elif label == 1:
        cValue_source.append('r')  # red
    elif label == 2:
        cValue_source.append('b')  # blue
    elif label == 3:
        cValue_source.append('m')  # magenta
    elif label == 4:
        cValue_source.append('y')  # yellow

for label in target_labels:
    if label == 0:
        cValue_target.append('g')  # green
    elif label == 1:
        cValue_target.append('r')  # red
    elif label == 2:
        cValue_target.append('b')  # blue
    elif label == 3:
        cValue_target.append('m')  # magenta
    elif label == 4:
        cValue_target.append('y')  # yellow

size_source = 120  # '+' 大小

size_target = 60  # 'D' 大小

print()
"""同时计算"""
tsne = TSNE(n_components=2, init='pca', early_exaggeration=30)
tsne.fit_transform(feature)

print()
plt.scatter(tsne.embedding_[:219*1, 0], tsne.embedding_[:219*1, 1], s=size_source, color='g', marker='+', label='S-1')
print('S-1.shape:', tsne.embedding_[:219*1, 0].shape)
plt.scatter(tsne.embedding_[1095:1095+97*1, 0], tsne.embedding_[1095:1095+97*1, 1], s=size_target, color='', marker='D', edgecolors='g', label='T-1')
print('T-1.shape:', tsne.embedding_[1095:1095+97*1, 0].shape)

print()
plt.scatter(tsne.embedding_[219*1:219*2, 0], tsne.embedding_[219*1:219*2, 1], s=size_source, color='r', marker='+', label='S-2')
print('S-2.shape:', tsne.embedding_[219*1:219*2, 0].shape)
plt.scatter(tsne.embedding_[1095+97*1:1095+97*2, 0], tsne.embedding_[1095+97*1:1095+97*2, 1], s=size_target, color='', marker='D', edgecolors='r', label='T-2')
print('T-2.shape:', tsne.embedding_[1095+97*1:1095+97*2, 0].shape)

print()
plt.scatter(tsne.embedding_[219*2:219*3, 0], tsne.embedding_[219*2:219*3, 1], s=size_source, color='b', marker='+', label='S-3')
print('S-3.shape:', tsne.embedding_[219*2:219*3, 0].shape)
plt.scatter(tsne.embedding_[1095+97*2:1095+97*3, 0], tsne.embedding_[1095+97*2:1095+97*3, 1], s=size_target, color='', marker='D', edgecolors='b', label='T-3')
print('T-3.shape:', tsne.embedding_[1095+97*2:1095+97*3, 0].shape)

print()
plt.scatter(tsne.embedding_[219*3:219*4, 0], tsne.embedding_[219*3:219*4, 1], s=size_source, color='m', marker='+', label='S-4')
print('S-4.shape:', tsne.embedding_[219*3:219*4, 0].shape)
plt.scatter(tsne.embedding_[1095+97*3:1095+97*4, 0], tsne.embedding_[1095+97*3:1095+97*4, 1], s=size_target, color='', marker='D', edgecolors='m', label='T-4')
print('T-4.shape:', tsne.embedding_[1095+97*3:1095+97*4, 0].shape)

print()
plt.scatter(tsne.embedding_[219*4:219*5, 0], tsne.embedding_[219*4:219*5, 1], s=size_source, color='y', marker='+', label='S-5')
print('S-5.shape:', tsne.embedding_[219*4:219*5, 0].shape)
plt.scatter(tsne.embedding_[1095+97*4:, 0], tsne.embedding_[1095+97*4:, 1], s=size_target, color='', marker='D', edgecolors='y', label='T-5')
print('T-5.shape:', tsne.embedding_[1095+97*4:, 0].shape)


plt.xticks([])
plt.yticks([])

plt.xlabel(r'$Component\ 1$', fontdict={'family': 'Times New Roman', 'size': 14})
plt.ylabel(r'$Component\ 2$', fontdict={'family': 'Times New Roman', 'size': 14})

# plt.legend(ncol=5, fancybox=True, fontsize=16, frameon=True)


plt.show()




