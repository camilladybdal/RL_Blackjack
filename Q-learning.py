from Blackjack import *
import math
import numpy as np
from utils import * 

class Q_learning_agent:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.Q_table = {}
        for i in range(4,22): #player sum
            for j in range(2,12): #dealer upcard
                for k in [True, False]: #soft hand
                    self.Q_table[(i,j,k)] = {}
                    for a in range(4):
                        self.Q_table[(i,j,k)][a] = 0 

    def get_maxQ(self, state):
        state = (state[0], state[1], state[2])
        best_action_val = - math.inf 
        best_action = 0
        for i in range(len(actions)):
            if self.Q_table[state][i] > best_action_val:
                best_action_val = self.Q_table[state][i]
                best_action = i
        return best_action_val, best_action 
    
    def get_action(self, state, new_hand, epsilon):
        state = (state[0], state[1], state[2])
        random_number = np.random.rand()
        no_double = [0,1,3]

        if random_number < epsilon:
            if new_hand == False:
                return actions[no_double[np.random.randint(0,3)]]
            action = np.random.randint(0, 4)
        else:
            if new_hand == False:
                best_action_val = - math.inf 
                for i in no_double:
                    if self.Q_table[state][i] > best_action_val:
                        best_action_val = self.Q_table[state][i]
                        action = i
            else: 
                _, action = self.get_maxQ(state)
        return actions[action]
    
    def update(self, old_state, new_state, action, reward, alpha, gamma):
        action = actions_to_numbers[action]

        Q = self.Q_table[old_state][action]
        
        if new_state[0] > 21:
            self.game_over = True
            maxQ_next = -1 
            Q_new = Q + alpha * (reward + gamma * maxQ_next - Q) 

        else:
            maxQ_next, _ = self.get_maxQ(new_state)
            Q_new = Q + alpha * (reward + gamma * maxQ_next - Q)

        self.Q_table[old_state][action] = Q_new 
    
def q_learning_loop(episodes):
    checkpoint = episodes // 10
    test_episodes = 1000000
    wins, losses, draws = 0, 0, 0
    eps_decay = 0.9999
    eps_min = 0.1
    epsilon = 0.9
    alpha = 0.09
    gamma = 0.85 
    rewards = []

    env = Blackjack()
    agent = Q_learning_agent(epsilon)

    #TRAINING
    for episode in range(episodes):
        if episode > episodes * 0.1:
            epsilon = max(eps_min, epsilon * eps_decay)
        if   episodes *0.6 < episode < episodes * 0.7:
            epsilon = 0.99

        state = env.reset() 
        state = (state[0], state[1], state[2])
        done = False

        while not done:
            state = env.get_state()
            state = (state[0], state[1], state[2])
            
            action = agent.get_action(state, env.new_hand, epsilon)

            new_state, reward, done = env.step(action) 
            new_state = (new_state[0], new_state[1], new_state[2])
            agent.update(state, new_state, action, reward, alpha, gamma)

        if episode % checkpoint == 0:
            save_Q_table(agent.Q_table)
            print("checkpoint saved: ", episode)
        
        rewards.append(reward)

    save_Q_table(agent.Q_table)
    print("checkpoint saved: ", episode)
    
    #TESTING
    rewards = 0  
    for episode in range(test_episodes):
        state = env.reset()
        state = (state[0], state[1], state[2])
        done = False

        while not done:
            action = agent.get_action(state, env.new_hand, epsilon)
            state, reward, done = env.step(action)
            rewards += reward
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        elif reward == 0:
            draws += 1  

    print("Test Wins: ", wins)
    print("Test Losses: ", losses)
    print("Test Draws: ", draws)
    print("avergae reward: ", rewards / test_episodes)

def play_random(episodes):
    wins, losses, draws = 0, 0, 0
    env = Blackjack()
    rewards = 0
    env.reset()
    for i in range(episodes):
        env.reset()
        done = False

        while not done:
            if env.new_hand:
                action = actions[np.random.randint(0,4)]
            else:
                possible_actions = [0, 1, 3]
                action = actions[possible_actions[np.random.randint(0,3)]]

            _, reward, done = env.step(action)       

        rewards += reward
            
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        elif reward == 0:
            draws += 1
    
    print("Random Wins: ", wins)
    print("Random Losses: ", losses)
    print("Random Draws: ", draws)
    print("avergae reward: ", rewards / episodes)

def play_basic(episodes):
    wins, losses, draws = 0, 0, 0
    env = Blackjack()
    env.reset()

    file = open("basic.csv", "r")
    strategy = []
    for line in file:
        strategy.append(line.split("\t"))

    rewards = 0

    for i in range(100000):
        env.reset()
        done = False

        while not done:
            player_hand, dealer_sum, soft = env.get_state()

            if not soft:
                action = strategy[player_hand-4][dealer_sum-2]
                action = action.strip('\n')
                if not env.new_hand and action == "double":
                    action = "hit"
            else:
                print("2")
                action = strategy[player_hand+6][dealer_sum-2]
                action = action.strip('\n')
                if not env.new_hand and action == "double" and player_hand < 17:
                    action = "hit"
                elif not env.new_hand and action == "double" and player_hand > 16:
                    action = "stand"

            print("action ", action)
  
            _, reward, done = env.step(action)       

        print("reward: ", reward)
        rewards += reward
            
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        elif reward == 0:
            draws += 1
    
    print("Basic Wins: ", wins)
    print("Basic Losses: ", losses)
    print("Basic Draws: ", draws)
    print("avergae reward: ", rewards / episodes)
