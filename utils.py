from Blackjack import actions
import tensorflow as tf
import numpy as np

def save_strategy(TrainNet, filename):
    with open(filename, 'w') as f:
        for i in range(4,22):
            f.write('\n')
            for j in range(2,12):
                for k in [True, False]:
                    for l in [True, False]:
                        f.write(str(i) + "," + str(j) + " ")
                        
                        states = tf.constant([(i,j,k,l)])
                        action = np.argmax(TrainNet.predict(states))

                        f.write("soft:" + str(k) + "new_hand:" + str(l) + " " + actions[action])
                        f.write(" ")