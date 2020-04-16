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


# 各参数 准确率 MMD变化趋势折线图
alpha_params = [0.01, 0.05, 0.1, 0.5, 1]
beta_params = [0.01, 0.02, 0.03, 0.04, 0.05]

beta_color = {0: 'k', 0.01: 'r', 0.02: 'g', 0.03: 'b', 0.04: 'c', 0.05: 'm'}  # beta参数与颜色映射字典

plt.figure(figsize=(10, 5), dpi=125)
for beta in beta_params:
    print('-'*10)
    acc = []  # 全部准确率
    acc_mean = []  # 后100次平均准确率

    color = beta_color[beta]

    for alpha in alpha_params:
        acc.append(np.load(r'A:\我的交大\Transfer Learning\Deep Transfer Learning\条件分布适配\a{} b{}\total_acc.npy'.format(alpha, beta)))
        print('alpha:{}, beta:{}, acc:'.format(alpha, beta), np.mean(acc[-1][-100:]))
        acc_mean.append(np.mean(acc[-1][-100:]))

    plt.plot(np.arange(0, len(acc_mean)), acc_mean, color=color, marker='o', markersize=6, label=r'$\beta = {}$'.format(beta))
    print('-'*10)
plt.xlabel(r'$Tradeoff\ parameter\ \alpha$', fontdict={'family': 'Times New Roman', 'size': 14})
plt.ylabel(r'$Accuracy$', fontdict={'family': 'Times New Roman', 'size': 14})
plt.xticks(np.arange(0, len(acc_mean)), alpha_params)
plt.legend()


plt.figure(figsize=(10, 5), dpi=125)
for beta in beta_params:
    print('-'*10)
    MMD = []  # 全部MMD
    MMD_mean = []  # 后100次平均MMD

    color = beta_color[beta]

    for alpha in alpha_params:
        MMD.append(np.load(r'A:\我的交大\Transfer Learning\Deep Transfer Learning\条件分布适配\a{} b{}\transfer_loss.npy'.format(alpha, beta)))
        print('alpha:{}, beta:{}, MMD:'.format(alpha, beta), np.mean(MMD[-1][-100:]))
        MMD_mean.append(np.mean(MMD[-1][-100:]))

    plt.plot(np.arange(0, len(MMD_mean)), MMD_mean, color=color, marker='o', markersize=6, label=r'$\beta = {}$'.format(beta))
    print('-'*10)
plt.xlabel(r'$Tradeoff\ parameter\ \alpha$', fontdict={'family': 'Times New Roman', 'size': 14})
plt.ylabel(r'$MMD$', fontdict={'family': 'Times New Roman', 'size': 14})
plt.xticks(np.arange(0, len(MMD_mean)), alpha_params)
plt.legend()

plt.show()






