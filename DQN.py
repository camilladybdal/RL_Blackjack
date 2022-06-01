import os
from ReplayMemory import *
from Blackjack import *
from utils import *
import tensorflow as tf
import matplotlib as plt
import math

def create_model(input_form, n_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_form),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_actions, activation='linear')
    ])
    model.summary()
    return model

class DQN:
    def __init__(self, input, n_actions):
        self.memory = ReplayMemory(1000)
        self.batch_size = 32
        self.min_replay_size = 1000
        self.n_actions = n_actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001)
        self.model = create_model(input, n_actions)
        self.loss_function = tf.losses.mean_squared_error

    def predict(self, states):
        return self.model.predict(states, verbose = 0)

    #Training function is inspired from professors's code in notebook. 
    def train(self, TargetNet, discount_factor):
        if len(self.memory.experience['state']) < self.min_replay_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.get_experience(self.batch_size)
        actions = np.array([actions_to_numbers[i] for i in actions])
        Q_next = TargetNet.predict(self.normalize_input(next_states, is_batch=True))
        max_Q_next = np.max(Q_next, axis=1)
        target_Q_val = (rewards + discount_factor * max_Q_next * (1 - dones))

        with tf.GradientTape() as tape:
            all_Q_val = self.model(self.normalize_input(states, is_batch=False))
            Q_val = tf.reduce_sum(all_Q_val * tf.one_hot(actions, self.n_actions), axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_function(target_Q_val, Q_val))
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def copy_weights(self, input_net):
        self.model.set_weights(input_net.model.get_weights())

    def get_action(self, observation, epsilon, new_hand):
        no_double = [0, 1, 3]
        if np.random.random() < epsilon:
            if new_hand:
                return actions[np.random.randint(0, self.n_actions)]
            else:
                return actions[no_double[np.random.randint(0,3)]]
        else:
            observation = np.array([observation])
            if new_hand:
                return actions[np.argmax(self.model(self.normalize_input(observation, is_batch=False)))]
            else:
                no_double = (self.model(self.normalize_input(observation, is_batch=False))).numpy()[0]
                valid_actions = [no_double[0], no_double[1], -math.inf, no_double[3]]
                return actions[np.argmax(valid_actions)]
        
    def normalize_input(self, state, is_batch = False):
        player_hand_range = 21
        dealer_hand_range = 11
        if is_batch:
            for i in range(self.batch_size):
                state[i][0] = state[i][0] / player_hand_range
                state[i][1] = state[i][1] / dealer_hand_range
            return np.array(state)
        else:
            return np.array([[state[0][0] / player_hand_range, state[0][1] / dealer_hand_range, state[0][2]]])
            
def dqn_train_loop(checkpoint_dir, episodes):
    CHECKPOINT_DIR = checkpoint_dir
    EPISODES = episodes
    rewards = 0
    copy_step = 100
    num_actions = 4
    num_states = 3
    input = (num_states,)
    gamma = 0.9
    eps_decay = 0.999
    eps_min = 0.1
    epsilon = 0.99

    env = Blackjack()
    TargetNet = DQN(input, num_actions)
    TrainNet = DQN(input, num_actions)

    for episode in range(EPISODES):
        epsilon = max(eps_min, epsilon * eps_decay)
        if (episode % 1000 == 0): print(f"Episode: {episode}\tEpsilon: {epsilon}")

        state = env.reset()
        while True:
            action = TrainNet.get_action(state, epsilon, env.new_hand)
            prev_state = state
            state, reward, done = env.step(action)

            experience = {'state': prev_state, 'action': action, 'reward': reward, 'next_state': state, 'done': done}
            TrainNet.memory.add_experience(experience)
            loss = TrainNet.train(TargetNet, gamma)
            if (done): break

        if episode % copy_step == 0:
            print("Copying weights from MainNet to TargetNet")
            TargetNet.copy_weights(TrainNet)

        if episode % 5000 == 0:
            print("average reward: ", rewards/(episode+1))
            print("loss: ", loss)
            print("Saved model")
            TrainNet.save_weights(os.path.join(CHECKPOINT_DIR, f"main_{episode}"))
            save_strategy(TrainNet, "strategy.csv")
        
        rewards += reward

    print("average reward: ", rewards/(episodes))
    TrainNet.save_weights(os.path.join(CHECKPOINT_DIR, f"main_{episode}"))
    save_strategy(TrainNet, "strategy.csv")

def dqn_test_loop(episodes):
    EPISODES = episodes
    rewards = 0 
    Qnet = DQN((3,), 4)
    latest = tf.train.latest_checkpoint("checkpoints/")
    Qnet.load_weights(latest)
    model = Qnet.model

    env = Blackjack()
    wins, losses, draws = 0, 0, 0

    for episodes in range(EPISODES):
        state = env.reset()
        state = np.array([state])

        while not env.game_over:
            action = actions[np.argmax(model(Qnet.normalize_input(state, is_batch=False)))]
            state, reward, _ = env.step(action)

        rewards += reward
        if reward >= 1:
            wins += 1
        elif reward < 0:
            losses += 1
        elif reward == 0:
            draws += 1

    save_strategy(model, "strategy.csv")
    print("wins: ", wins)
    print("losses: ", losses)
    print("draws: ", draws)
    print("average reward: ", rewards/EPISODES)
