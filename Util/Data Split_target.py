import numpy as np
from scipy.io import loadmat


def data_process(paths, example_size):
    num_example = 304

    datas = []
    A = []
    for i in range(len(paths)):  # 每一个舰船
        datas.append(loadmat(paths[i]))
        datas[i] = list(datas[i]['y'].flatten())

        # 数据切片
        j = 0
        while j + example_size < len(datas[i]):
            A.append(datas[i][j:j + example_size])
            j += example_size

    A = np.array(A)  # 将列表转换为 nd array
    np.random.shuffle(A)

    A = A[:num_example]

    print('A.shape=', A.shape)
    return A


fs = 52734
example_size = 2000

root_path = r'A:\我的交大\Transfer Learning\数据\源A -目标A（滤波、无降采样）'

paths_target_1 = [root_path + r'\33.mat']

paths_target_2 = [root_path + r'\30.mat']

paths_target_3 = [root_path + r'\7.mat']

paths_target_4 = [root_path + r'\24.mat']

paths_target_5 = [root_path + r'\20.mat']


# T1
print('T1 start!')
T1 = data_process(paths_target_1, example_size)
print('T1 over！')
print()
np.save('T1.npy', T1)

# T2
print('T2 start!')
T2 = data_process(paths_target_2, example_size)
print('T2 over！')
print()
np.save('T2.npy', T2)

# T3
print('T3 start!')
T3 = data_process(paths_target_3, example_size)
print('T3 over！')
print()
np.save('T3.npy', T3)

# T4
print('T4 start!')
T4 = data_process(paths_target_4, example_size)
print('T4 over！')
print()
np.save('T4.npy', T4)

# T5
print('T5 start!')
T5 = data_process(paths_target_5, example_size)
print('T5 over！')
print()
np.save('T5.npy', T5)
