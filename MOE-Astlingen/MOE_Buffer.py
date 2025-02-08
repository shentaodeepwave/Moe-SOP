
import numpy as np
import pandas as pd
import scipy.signal
import torch

def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class DQNBuffer:
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer =[]
        self.observation_next_buffer = []
        self.action_buffer = []
        self.advantage_buffer = np.zeros(size, dtype=np.float16)
        self.reward_buffer = []
        self.return_buffer = np.zeros(size, dtype=np.float16)
        self.value_buffer = np.zeros(size, dtype=np.float16)
        
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, observation_next):
        self.observation_buffer.append(observation)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.observation_next_buffer.append(observation_next)
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        return (
            self.observation_buffer,
            self.action_buffer,
            self.observation_next_buffer,
            self.reward_buffer,
            0,
        )

