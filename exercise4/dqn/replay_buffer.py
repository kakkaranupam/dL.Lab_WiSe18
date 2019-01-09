from collections import namedtuple, deque
import random
import numpy as np
import os
import gzip
import pickle

class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self):
        #self.memory = deque(maxlen=10000)
        self._data = namedtuple("ReplayBuffer", field_names=["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])
    
    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)
        #e = self._data(state, action, reward, next_state, done)
        #self.memory.append(e)

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones    
        #experiences = random.sample(self.memory, k=batch_size)

        #states = np.vstack([e.state for e in experiences if e is not None])
        #actions = np.vstack([e.action for e in experiences if e is not None])
        #rewards = np.vstack([e.reward for e in experiences if e is not None])
        #next_states = np.vstack([e.next_state for e in experiences if e is not None])
        #dones = np.vstack([e.done for e in experiences if e is not None])
        #return states, actions, next_states, rewards, dones
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
