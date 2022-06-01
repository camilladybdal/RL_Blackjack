import numpy as np
from collections import deque

class ReplayMemory:
    def __init__ (self, max_length):
        self.max_length = max_length
        self.experience = {
            'state' : deque(maxlen=max_length),
            'action' : deque(maxlen=max_length),
            'reward' : deque(maxlen=max_length),
            'next_state' : deque(maxlen=max_length),
            'done' : deque(maxlen=max_length)
        }

    def add_experience(self, new_experience):
        for key, value in new_experience.items():
            self.experience[key].append(value)

    def get_experience(self, batch_size):
        indices = np.random.randint(0, len(self.experience['state']), batch_size) 
        states = np.array([self.experience['state'][index] for index in indices])
        actions = np.array([self.experience['action'][index] for index in indices])
        rewards = np.array([self.experience['reward'][index] for index in indices])
        next_states = np.array([self.experience['next_state'][index] for index in indices])
        dones = np.array([self.experience['done'][index] for index in indices])
        
        return states, actions, rewards, next_states, dones