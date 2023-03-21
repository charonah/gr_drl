import os
import time

import numpy as np
from numpy import inf


'''
author: Zihao FANG

input: action(obs) , state_recognition,network, freq, accumulated_q
output: accumulated_q,  result.txt

some edit was in TD3_agent.py & velodyne_env.py
'''

def recogniton_observability(network, action, state, freq, accumulated_q):

    Qvalues = []

    # noise obs:0.5:4\\0.3:6\\0.1:10
    if freq % 10 == 0:
        action = (action + np.random.normal(0, 1, size=2)).clip(-1, 1)

    for state_ in state:
        Qvalue = network.get_Qvalue(np.array(action), np.array(state_))
        Qvalues.append(Qvalue)  

    Qvalues = np.array(Qvalues)

    # partial obs:1:1\\0.7:2\\0.5:3\\0.3:4\\0.1:10
    if freq % 1 == 0:
        accumulated_q += Qvalues 
        # print(np.argmax(accumulated_q, axis=0))

    return accumulated_q


def writter_file(recognition_episode_results, obs_type, file = None):
    accuracy, precision, recall, fscore = calculate_all_metrics(recognition_episode_results)
    print('Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
    if file:
        file = open(file, 'a')
        file.write(f"******  Results for {obs_type} ******\n")
        file.write(f"# {obs_type} \n")

        file.write(f'#OBS\t Acc\t Prec\t Rec\t F-S\n')
        print('OBS:', obs_type, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
        file.write(f'{obs_type}\t{accuracy:.2f}\t{precision:.2f}\t{recall:.2f}\t{fscore:.2f}\n') 



'''
TP True Positive 实际为垃圾邮件你预测为垃圾邮件，
FP False Positive 实际不是垃圾邮件你预测为垃圾邮件
TN True Negative 实际为垃圾邮件你预测为不是垃圾邮件，
FN False Negative 实际不是垃圾邮件你预测为不是垃圾邮件。
result:accumulated_q
'''
def run_domain_metrics(real_goal, result):
    
    domain_results = dict()
    keys = ['TP', 'FP', 'FN', 'TN', 'len']
    for key in keys:
        domain_results[key] = 0
    # print(domain_results)

    tp, fn, fp, tn = measure_confusion(result, real_goal)      
    domain_results['TP'] += tp
    domain_results['FP'] += fp
    domain_results['FN'] += fn
    domain_results['TN'] += tn
    domain_results['len'] += 5

    # print(domain_results)
    return domain_results



def measure_confusion(result, real_goal):
    if np.argmax(result, axis=0) == real_goal:
        prediction = True
    else:
        prediction = False

    ranking = np.sort(result, axis=0) # sort by acc_Qvalue :low to high

    head = ranking[-1]   #the biggest value
    tail = ranking[0:-1]
    fn = int(not prediction)
 
    fp = 0
    tn = 0
    if prediction:       
        for goal_value in tail:
            if goal_value == head:
                fp += 1
            else:
                tn += 1    
    else:
        fp = 1
        for goal_value in tail[:-1]:
            if goal_value == head:
                fp += 1
            else:
                tn += 1

    #      tp               fn                   fp  tn       
    return int(prediction), fn, fp, tn


# obs_metric : domain_results(one_step_result)
def calculate_all_metrics(obs_metrics):
    accuracy = 0
    precision = 0
    recall = 0
    fscore = 0
    accuracy = (obs_metrics['TP'] + obs_metrics['TN']) / obs_metrics['len']
    precision = obs_metrics['TP'] / (obs_metrics['TP'] + obs_metrics['FP'])
    recall = obs_metrics['TP'] / (obs_metrics['TP'] + obs_metrics['FN'])
    if precision + recall != 0:
        fscore = (2 * precision * recall) / (precision + recall)
    else:
        fscore = 0
    return accuracy, precision, recall, fscore

    