import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_AUC_SCORE = 1


class CoMetEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(CoMetEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_AUC_SCORE)

        # Actions
        # self.action_space = spaces.Box()

        # 1296 batches and 2294 meta-features per batch
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1296, 2294), dtype=np.float16)

    def _next_observation(self):
        frame = np.array([

        ])

        obs = np.append(frame, [[

        ]], axis=0)

        return obs

    def _take_action(self, action):
        pass


    def step(self, action):
        pass
        #return obs, reward, done, {}

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass
