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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.flatten2 = nn.Linear(in_features=2480, out_features=256)

        self.flatten3 = nn.Linear(in_features=256, out_features=5)

    # 前向传播
    def forward(self, source, target):
        bs_source = source.shape[0]
        bs_target = target.shape[0]

        source = F.leaky_relu(self.conv1(source))
        target = F.leaky_relu(self.conv1(target))

        source = F.leaky_relu(self.conv2(self.pool1(source)))
        target = F.leaky_relu(self.conv2(self.pool1(target)))

        source = (self.pool2(source)).view(bs_source, -1)
        target = (self.pool2(target)).view(bs_target, -1)

        source = F.leaky_relu(self.flatten2(source))
        target = F.leaky_relu(self.flatten2(target))

        print('source feature:', source.shape)
        print('target feature:', target.shape)

        np.save('条件分布适配_source.npy', source.cpu().detach().numpy())
        np.save('条件分布适配_target.npy', target.cpu().detach().numpy())
        print('特征已保存！')

        source_clf = self.flatten3(source)
        target_clf = self.flatten3(target)

        return source_clf, target_clf

    # 预测
    def predict(self, source):
        bs = source.shape[0]

        source = F.leaky_relu(self.conv1(source))
        source = F.leaky_relu(self.conv2(self.pool1(source)))
        source = (self.pool2(source)).view(bs, -1)
        source = F.leaky_relu(self.flatten2(source))

        source_clf = self.flatten3(source)

        return source_clf


def accuracy_m(net, data_iter):
    net.eval()
    with torch.no_grad():
        num_acc = 0
        num_examples = 0
        matrix = torch.zeros((5, 5))
        for X, y in data_iter:

            predicts = net.predict(X)

            num_acc += (predicts.argmax(dim=1, keepdim=True) == y.view(-1, 1)).float().sum().item()

            num_examples += X.size()[0]

            for i in range(0, X.size(0)):

                matrix[y[i].item(), predicts.argmax(dim=1, keepdim=True)[i].item()] += 1

        return num_acc/num_examples, matrix


"""设置超参数"""
batch_size = 32

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU数量：', torch.cuda.device_count())
    print('当前GPU索引号:', torch.cuda.current_device())
    print('根据索引号查看GPU名字：', torch.cuda.get_device_name(0))

# 加载源域
path_source = ['S1.npy', 'S2.npy', 'S3.npy', 'S4.npy', 'S5.npy']

for i in [0, 1, 2, 3, 4]:  # 读入数据和标签转换成tensor
    print('-' * 10)
    if i == 0:
        source_data = torch.tensor(np.load(path_source[i]), dtype=torch.float32)
        # source_data = Z_Score(source_data)
        source_data = source_data.view(source_data.shape[0], 1, 1, source_data.shape[1])
        print('source_data_{}.shape'.format(i), source_data.shape)
        source_labels = torch.zeros((source_data.shape[0], 1), dtype=torch.long)
        print('source_labels_{}.shape'.format(i), source_labels.shape)
    else:
        temp_data = torch.tensor(np.load(path_source[i]), dtype=torch.float32)
        # temp_data = Z_Score(temp_data)
        temp_data = temp_data.view(temp_data.shape[0], 1, 1, temp_data.shape[1])
        print('source_data_{}.shape'.format(i), temp_data.shape)
        temp_labels = torch.zeros((temp_data.shape[0], 1), dtype=torch.long) + i
        print('source_labels_{}.shape'.format(i), temp_labels.shape)

        source_data = torch.cat([source_data, temp_data], dim=0)
        source_labels = torch.cat([source_labels, temp_labels], dim=0)
    print('-' * 10)

# 打印读取后source的数据和标签shape
print('-' * 10)
print('source.shape:', source_data.shape)
print('source_labels.shape', source_labels.shape)
print('-' * 10)

# 加载目标域
path_target = ['T1.npy', 'T2.npy', 'T3.npy', 'T4.npy', 'T5.npy']

for i in [0, 1, 2, 3, 4]:  # 读入数据和标签转换成tensor
    print('-' * 10)
    if i == 0:
        target_data = torch.tensor(np.load(path_target[i]), dtype=torch.float32)
        # target_data = Z_Score(target_data)
        target_data = target_data.view(target_data.shape[0], 1, 1, target_data.shape[1])
        print('target_data_{}.shape'.format(i), target_data.shape)
        target_labels = torch.zeros((target_data.shape[0], 1), dtype=torch.long)
        print('target_labels_{}.shape'.format(i), target_labels.shape)
    else:
        temp_data = torch.tensor(np.load(path_target[i]), dtype=torch.float32)
        # temp_data = Z_Score(temp_data)
        temp_data = temp_data.view(temp_data.shape[0], 1, 1, temp_data.shape[1])
        print('target_data_{}.shape'.format(i), temp_data.shape)
        temp_labels = torch.zeros((temp_data.shape[0], 1), dtype=torch.long) + i
        print('target_labels_{}.shape'.format(i), temp_labels.shape)

        target_data = torch.cat([target_data, temp_data], dim=0)
        target_labels = torch.cat([target_labels, temp_labels], dim=0)
    print('-' * 10)

# 打印读取后target的数据和标签shape
print('-' * 10)
print('target.shape:', target_data.shape)
print('target_labels.shape', target_labels.shape)
print('-' * 10)

"""打印网络结构"""
net = Net()
print('-' * 10)
print(net)
print('-' * 10)

"""创建数据生成器"""
# source_dataset = Data.TensorDataset(source_data, source_labels)
# source_iter = Data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
#
# target_dataset = Data.TensorDataset(target_data, target_labels)
# target_iter = Data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True)


source_dataset = Data.TensorDataset(source_data.to(device), source_labels.to(device))
source_iter = Data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True)

target_dataset = Data.TensorDataset(target_data.to(device), target_labels.to(device))
target_iter = Data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

net.to(device)

# 加载模型参数
PATH = 'Last_条件分布适配_params.pth'
net.load_state_dict(torch.load(PATH))

acc, matrix = accuracy_m(net, target_iter)
print()
print('准确率：', acc)
print('混淆矩阵：')
print(matrix)
print()

net(source_data.to(device), target_data.to(device))


