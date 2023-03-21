import numpy as np
import math
import os
import time
from recognition import calculate_all_metrics


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



# a= [(2, -99.21875), (1, -0.0), (3, -0.0)]

# print(a[:-1])

# result = {'TP': 1, 'FP': 0, 'FN': 0, 'TN': 4, 'len': 5}
# result1 = {'TP': 1, 'FP': 0, 'FN': 0, 'TN': 4, 'len': 5}
# keys = ["TP", "FP", "FN", "TN", "len"]
# result2 = dict()
# for key in keys:
#         result2[key] =result1[key] + result[key]

# print(result2)

# accuracy, precision, recall, fscore = calculate_all_metrics(result)
# print
# goal_set = np.argmax(goal_set, axis=0)
# real_goal_index = [3]

# print(goal_set)
# print(type(goal_set))
# print(real_goal_index == goal_set)
x_set =[-0.5747037629520912, -1.182659163371806, 1.5173417612281082, 1.5108878245215633]
y_set =[4.246934303446224, -1.3415226805021918, 4.457259286108737, -4.4323475800689796]
           
print(x_set[1],y_set[1],type(x_set[1]))


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