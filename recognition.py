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
    # partial obs:1:1\\0.7:2\\0.5:3\\0.3:4\\0.1:10
    if freq % 1 == 0:
        # noise obs:0.5:4\\0.3:6\\0.1:10
        # if freq % 10 == 0:
        #     action = (action + np.random.normal(0, 0.3, size=2))
        #     # action = np.random.normal(0, 1, size=2).clip(-1, 1)
    
        for state_ in state:
            Qvalue = network.get_Qvalue(np.array(action), np.array(state_))
            Qvalues.append(Qvalue)  

        Qvalues = np.array(Qvalues)
        accumulated_q += Qvalues 
    # print(np.argmax(accumulated_q, axis=0))
    return accumulated_q

def recogniton_observability_dict(network, action, state, freq, accumulated_q_dict, partial):
    for key in partial:  
        Qvalues = []
        # partial obs:1:1\\0.7:2\\0.5:3\\0.3:4\\0.1:10
        if freq % key == 0:

            for state_ in state:
                Qvalue = network.get_Qvalue(np.array(action), np.array(state_))
                Qvalues.append(Qvalue)  

            Qvalues = np.array(Qvalues) 
            accumulated_q_dict[str(key)] += Qvalues 
    # print(accumulated_q_dict)
    return accumulated_q_dict


def recogniton_observability_loss(network, action, state, freq, accumulated_q_loss_dict, sample, SAMPLE):
    for key in SAMPLE:
        Qvalues = []
        if freq in sample[str(key)]:
            for state_ in state:
                Qvalue = network.get_Qvalue(np.array(action), np.array(state_))
                Qvalues.append(Qvalue)  

            Qvalues = np.array(Qvalues)
            accumulated_q_loss_dict[str(key)] += Qvalues 
 
    return accumulated_q_loss_dict

def recogniton_observability_noise(network,
                                action, 
                                state,
                                freq, 
                                accumulated_q_noise, 
                                db, 
                                partial,
                                GAUSSIAN = False,
                                PASSION = False,
                                LAPLACE = False):
    # 生成不同信噪比的噪声数据
    for key in db:
        Qvalues = []
        state = np.array(state)
        # 生成均值为0、标准差为1的高斯噪声数据，形状与原始数据相同
        if GAUSSIAN: 
            gaussian_noise_action = np.random.normal(0, 1, action.shape)
            gaussian_noise_state = np.random.normal(0, 1, state.shape)
            noise_action = gaussian_noise_action * 10 ** (-key / 20)
            noise_state = gaussian_noise_state * 10 ** (-key / 20)
        if PASSION:
            poisson_noise_action = np.random.poisson(2, action.shape)
            poisson_noise_state = np.random.poisson(2, state.shape)
            noise_action = poisson_noise_action * 10 ** (-key / 20)
            noise_state = poisson_noise_state * 10 ** (-key / 20)
        if LAPLACE:
            std_action = np.std(action) / (10 ** (key / 20))
            std_state = np.std(state) / (10 ** (key / 20))
            laplace_noise_action = np.random.laplace(scale=std_action, size=action.shape)
            laplace_noise_state = np.random.laplace(scale=std_state, size=state.shape)
            noise_action = action + laplace_noise_action
            noise_state = state + laplace_noise_state

        # 将噪声数据添加到输入数据中
        noisy_action = action + noise_action
        noisy_state = state + noise_state
        state = list(state)
        if freq % partial == 0:
            action = noisy_action
            state = noisy_state
            #     # action = np.random.normal(0, 1, size=2).clip(-1, 1)
        
            for state_ in state:
                Qvalue = network.get_Qvalue(np.array(action), np.array(state_))
                Qvalues.append(Qvalue)  

            Qvalues = np.array(Qvalues)
            accumulated_q_noise[str(key)] += Qvalues 
    # print(np.argmax(accumulated_q, axis=0))
    return accumulated_q_noise


def writter_file(recognition_episode_results, obs_type, file = None):
    accuracy, precision, recall, fscore = calculate_all_metrics(recognition_episode_results)
    print('Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
    if file:
        file = open(file, 'a')
        file.write(f"******  Results for {obs_type} ******\n")
        file.write(f"# {obs_type} \n")

        file.write(f'#OBS\t Acc\t Prec\t Rec\t F-S\n')
        print('OBS:', obs_type, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
        file.write(f'{obs_type}\t{accuracy:.3f}\t{precision:.3f}\t{recall:.3f}\t{fscore:.3f}\n') 

def writter_time_file(time_array, key, file = None):
    if file:
        file = open(file, 'a')
        file.write(f"******  Results for dynamics {key} ******\n")
        arv_time = sum(time_array)/len(time_array)
        file.write(f'{key}\t{arv_time:.3f}\n') 

'''
TP True Positive 
FP False Positive 
TN True Negative 
FN False Negative 
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

    