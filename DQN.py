from ReplayMemory import *
from Blackjack import *
from Model import *
from utils import *
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
import logging

class DQN:
    def __init__(self, input, n_actions):
        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.min_replay_size = 100
        self.n_actions = n_actions

        self.optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001)
        self.model = create_model(input, n_actions)
        self.loss_function = tf.losses.mean_squared_error

    def predict(self, states):
        return self.model.predict(states, verbose = 0)

    #TODO: skriv om
    def train(self, TargetNet, discount_factor):
        if len(self.memory.experience['state']) < self.min_replay_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.get_experience(self.batch_size)
        actions = np.array([actions_to_numbers[i] for i in actions])

        next_Q_values = TargetNet.predict(next_states) 
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + discount_factor * max_next_Q_values * (1 - dones))
        mask = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def copy_weights(self, TargetNet):
        for v1, v2 in zip(self.model.trainable_variables, TargetNet.model.trainable_variables):
            v2.assign(v1.numpy())

    def get_action(self, observation, epsilon): #returns action in number
        if np.random.random() < epsilon:
            return actions[np.random.randint(0, self.n_actions)]
        else:
            observation = np.array([observation])
            return actions[np.argmax(self.model(observation))]

#######################################################################################################

def dqn_train_loop(checkpoint_dir, episodes):
    CHECKPOINT_DIR = checkpoint_dir
    EPISODES = episodes
    rewards = np.empty(EPISODES)
    losses =  np.empty(EPISODES)

    copy_step = 25
    num_actions = 4
    num_states = 4
    input = (num_states,)
    iterations = 0
    gamma = 0.95
    eps_decay = 0.9999
    eps_min = 0.1
    epsilon = 0.99

    env = Blackjack()
    TargetNet = DQN(input, num_actions)
    TrainNet = DQN(input, num_actions)

    for episode in range(EPISODES):
        print("Playing new episode: ", episode)
        if episode < 0.2*EPISODES:
            epsilon = max(eps_min, epsilon * eps_decay)

        state = env.reset()

        while not env.game_over:
            action = TrainNet.get_action(state, epsilon)

            prev_state = state
            state, reward, done = env.step(action)

            experience = {'state': prev_state, 'action': action, 'reward': reward, 'next_state': state, 'done': done}
            TrainNet.memory.add_experience(experience)
            loss = TrainNet.train(TargetNet, gamma)
            
            iterations += 1
            if iterations % copy_step == 0:
                TargetNet.copy_weights(TrainNet)
            
        rewards[episode] = reward
        if loss == None: loss = 0
        losses[episode] = loss

        if episode % 500 == 0:
            print("saved model:" , episode)
            print("average reward: ", np.mean(rewards))
            print("loss: ", loss.numpy())

            TrainNet.model.save("my_model")
            save_strategy(TrainNet, "strategy.csv")

    print("average reward: ", np.mean(rewards))
    print("average loss: ", np.mean(loss))
    TrainNet.save_weights(CHECKPOINT_DIR )
    save_strategy(TrainNet, "strategy.csv")

    plt.plot(rewards)
    plt.label("Rewards")
    plt.show()

def dqn_test_loop(episodes):
    EPISODES = episodes

    model = keras.models.load_model("my_model")
    env = Blackjack()
    
    rewards = np.empty(EPISODES)
    wins, losses, draws = 0, 0, 0

    for episodes in range(EPISODES):
        state = env.reset()
        state = np.array([state])

        while not env.game_over:
            action = actions[np.argmax(model(state))]
            state, reward, _  = env.step(action)
            state = np.array([state])
        
        if reward >= 1:
            wins += 1
        elif reward < 0:
            losses += 1
        elif reward == 0:
            draws += 1

        rewards[episodes] = reward

    print("wins: ", wins)
    print("losses: ", losses)
    print("draws: ", draws)
    print("average reward: ", np.mean(rewards))    