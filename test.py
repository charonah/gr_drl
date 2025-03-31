import numpy as np
import math
import os
import time
from recognition import calculate_all_metrics
import psutil
import random

# goal_set = [[4,4], [0,4], [-4,3], [4.5, -2.5],[-4,-2]]
# distance_recognition = []
# for x, y  in goal_set:
#     a = np.linalg.norm([0 - x, 4 - y])
#     distance_recognition.append(a)
# print(distance_recognition)


# a = [[1], [2], [3]]
# b = []
# i = 0
# for x in a:
#     b = x 
#     i += 1

# print(b)

goal_set = [[4], [0], [-4], [4.5],[-4]]
goal_set = np.array(goal_set)
rank = np.sort(goal_set, axis=0)

print(np.sort(goal_set, axis=0))

head = rank[-1]   #the biggest value
tail = rank[0:-1]

for goal_value in tail:
        print(goal_value == [4])
print(tail)
print(tail[:-1])


# 信号的标准差
signal_std = 1.0

# 不同信噪比的噪声标准差
snr_db = [0, 5, 10]  # 0 dB, 5 dB, 10 dB
snr = [10 ** (i / 10.0) for i in snr_db]  # 转换为线性信噪比
noise_std = [signal_std / s for s in snr]  # 计算不同信噪比下的噪声标准差

# 生成高斯白噪声信号
N = 1000  # 信号长度
for i, std in enumerate(noise_std):
    noise = np.random.normal(0, std, size=(N,))
    # 添加信号
    signal = np.sin(2 * np.pi * 10 * np.arange(N) / N)
    noisy_signal = signal + noise

    # 计算信噪比
    snr_out = 10 * np.log10(np.sum(signal ** 2) / np.sum(noise ** 2))

    print(f"信噪比: {snr_db[i]} dB,生成噪声的标准差: {std:.4f}，计算的信噪比: {snr_out:.4f}")

# neicunzhanyong 

# import psutil
# process = psutil.Process()
# process_memory_start = process.memory_info().rss / 1024 / 1024

# process_memory_end = process.memory_info().rss / 1024 / 1024
# process_memory_usage = process_memory_end - process_memory_start
# print('Memory usage:', process_memory_usage, 'MB')


# # 输入数据
# input_data = np.random.uniform(-1, 1, size=2)

# # 信噪比列表
# snr_list = [10, 5, 2]

# # 生成不同信噪比的噪声数据
# for snr in snr_list:
#     # 计算噪声功率
#     noise_power = np.var(input_data) / (10 ** (snr / 10))
#     # 生成噪声数据
#     noise_data = np.random.normal(scale=np.sqrt(noise_power), size=2)
#     # 将噪声数据添加到输入数据中
#     noisy_data = input_data + noise_data
# #     print(noisy_data)
#     # 输出信噪比和噪声数据的标准差
#     print(f"SNR={snr}, Noise STD={np.std(noise_data)}")


# 生成原始数据
x = np.random.uniform(-1, 1, (2, 1))
print(x)
# 生成均值为0、标准差为1的高斯噪声数据，形状与原始数据相同
gaussian_noise = np.random.normal(0, 1, x.shape)

# 按照不同的信噪比，生成不同比例的高斯噪声数据
snr_db = [10, 5, 0, -5, -10]  # 信噪比(dB)分别为10、5、0、-5、-10
for db in snr_db:
    noise = gaussian_noise * 10 ** (-db / 20)
    noisy_signal = x + noise
    print("SNR(dB)={}, Noise Variance={:.2f}, SNR={:.2f}".format(
        db, noise.var(), 10 * np.log10(np.mean(x ** 2) / noise.var())))
    print(noisy_signal)


# PARTIAL_OBS = [10, 8, 6, 4, 1] 
# for obs in PARTIAL_OBS:
#     print(type(obs))

# accumulated_q_dict = dict()
# PARTIAL_OBS = [10, 8, 6, 4, 1] 
# for key in PARTIAL_OBS:
#     accumulated_q_dict[str(key)] = None
#     print(accumulated_q_dict[str(key)])

# print((1%20))

# accumulated_q_loss_dict = dict()
# sample_dict = dict()
# SAMPLE = [8, 15, 45, 75]
# for key in SAMPLE:
#     accumulated_q_loss_dict[str(key)] = 0
#     sample_dict[str(key)] = np.random.randint(1, 105, size=key).tolist()
# print(sample_dict)




# ti = [9.54121994972229, 9.955924987792969, 7.007209300994873]
# if len(ti):
#         print(sum(ti)/len(ti))
# for i in range(10):
#         action = np.random.normal(0, 1, size=2).clip(-1, 1)
#         print(action)
# a= [(2, -99.21875), (1, -0.0), (3, -0.0)]

# print(a[:-1])

# result = {'TP': 1, 'FP': 0, 'FN': 0, 'TN': 4, 'len': 5}
# result1 = {'TP': 1, 'FP': 0, 'FN': 0, 'TN': 4, 'len': 5}
# # keys = ["TP", "FP", "FN", "TN", "len"]
# # result2 = dict()
# # for key in keys:
# #         result2[key] =result1[key] + result[key]

# # print(result2)

# # accuracy, precision, recall, fscore = calculate_all_metrics(result)
# # print
# # goal_set = np.argmax(goal_set, axis=0)
# # real_goal_index = [3]

# # print(goal_set)
# # print(type(goal_set))
# # print(real_goal_index == goal_set)
# x_set =[-0.5747037629520912, -1.182659163371806, 1.5173417612281082, 1.5108878245215633]
# y_set =[4.246934303446224, -1.3415226805021918, 4.457259286108737, -4.4323475800689796]
           
# print(x_set[1],y_set[1],type(x_set[1]))


# domain_results = dict()

# for obs in range(5):
#         domain_results[str(obs)] = dict()

# for key in domain_results:
#         domain_results[key]['TP'] = 0
#         domain_results[key]['FP'] = 0
#         domain_results[key]['FN'] = 0
#         domain_results[key]['TN'] = 0
#         domain_results[key]['len'] = 0
        
# domain_results = dict()
# key = ["TP", "FP", "FN", "TN", "len"]
# for obs in key:
#         domain_results[obs] = dict()
# print(domain_results)

# distance = np.linalg.norm([0 - 4, 4 - 4])
# print(distance)

        #     [self.odom_x - self.real_goal_x, self.odom_y - self.real_goal_y]