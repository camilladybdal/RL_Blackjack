import numpy as np

class ReplayMemory:
    def __init__ (self, max_length):
        self.max_length = max_length
        self.experience = {
            'state' : [],
            'action' : [],
            'reward' : [],
            'next_state' : [],
            'done' : []
        }

    def add_experience(self, experience):
        if len(self.experience['state']) > self.max_length:
            for key in self.experience:
                self.experience[key].pop(0)
        for key, value in experience.items():
            self.experience[key].append(value)

    def get_experience(self, batch_size):
        indices = np.random.randint(0, len(self.experience['state']), batch_size) 
        states = np.array([self.experience['state'][index] for index in indices])
        actions = np.array([self.experience['action'][index] for index in indices])
        rewards = np.array([self.experience['reward'][index] for index in indices])
        next_states = np.array([self.experience['next_state'][index] for index in indices])
        dones = np.array([self.experience['done'][index] for index in indices])
        
        return states, actions, rewards, next_states, dones