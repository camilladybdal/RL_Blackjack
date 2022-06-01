from Blackjack import actions
import tensorflow as tf
import numpy as np
import math
from DQN import *

def save_strategy(TrainNet, filename):
    with open(filename, 'w') as f:
        for i in range(4,22):
            f.write('\n')
            for j in range(2,12):
                for k in [True, False]:
                    f.write(str(i) + "," + str(j) + "," + str(k)+ " ")                    
                    states = np.array([i, j, k])
                    states = states.reshape(1,3)
                    action = np.argmax(TrainNet.predict(states))
                    if k == True and i < 12:
                        f.write('-')
                    else:
                        f.write(actions[action])
                    f.write(" ")

def save_Q_table(Q_table):
    with open('q_table_values.txt', 'w') as f:
        for i in range(4,22):
            f.write('\n')
            for j in range(2,12):
                for k in [True, False]:
                    f.write(str(i) + "," + str(j) + "," + str(k) + ":")
                    for a in range(4):
                        f.write(str(Q_table[(i,j,k)][a]))
                        if a !=2:
                            f.write(",")
                    f.write(" ")

    with open('q_table_actions.txt', 'w') as f:
        for i in range(4,22):
            f.write('\n')
            for j in range(2,12):
                for k in [True, False]:
                    best_action_val = - math.inf
                    for l in range(4):
                        if Q_table[(i,j,k)][l] > best_action_val:
                            best_action_index = l 
                            best_action_val = Q_table[(i,j,k)][l]
                    best_action = actions[best_action_index]
                    if k == True and i < 12:
                        best_action = '-'
                    f.write(str(i) + "," + str(j) + "," + str(k) + ":" +  best_action + "   ")
    f.close()