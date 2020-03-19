import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import meta_model_for_java_csv_v2 as meta_model
import subprocess

MAX_AUC_SCORE = 1
BATCH_CANDIDATES = 1296

class CoMetEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, dataset_arff, exp_id, modelFiles):
        super(CoMetEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_AUC_SCORE)
        self.reward = 0
        self.dataset = dataset_arff
        self.exp = exp_id
        self.file_prefix = str(exp_id) + "_" + dataset_arff[:-5] + "_"
        self.modelFiles = modelFiles
        self.iteration = 0
        self.selected_batch_id = -2
        self.df_col_list = df.columns

        # Actions
        self.action_space = spaces.Discrete(BATCH_CANDIDATES)
        # self.action_space = spaces.Discrete(1)

        # 1296 batches and 2233 meta-features per batch
        # high = np.array([np.inf]*N_OBS_DIM) --> self.observation_space = spaces.Box(-high, high)
        self.observation_space = spaces.Box(
            low=-exp_id, high=exp_id, shape=(BATCH_CANDIDATES, len(self.df_col_list)), dtype=np.float16)

    def _next_observation(self):
        print("iteration: {}".format(self.iteration))
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration",
                         self.file_prefix, str(self.selected_batch_id), str(self.iteration), str(self.exp)])
        score_ds_f, instance_batch_f, rewards, meta_f = self.meta_features_process(
            self.modelFiles, self.file_prefix, self.iteration, self.dataset)
        not_in_meta_f = list(set(self.df_col_list) - set(meta_f.columns))
        # ToDo: debug why the shape here is 2234 and not 2233 - is there a duplication?
        more_cols = {}
        for col in not_in_meta_f:
            if col not in more_cols:
                more_cols[col] = -1.0
        meta_f = meta_f.assign(**more_cols)
        meta_f = meta_f[self.df_col_list]
        meta_f = meta_f.loc[:, ~meta_f.columns.duplicated()]
        self.df = meta_f
        return meta_f, rewards

    # ToDo: implement the selection of the batch id
    def _take_action(self, action):
        print("action - take action: {}".format(action))
        self.selected_batch_id = random.randint(0, BATCH_CANDIDATES - 1)

    def step(self, action):
        self.iteration += 1
        self._take_action(action)
        obs, rewards = self._next_observation()
        print("Selected batch id: {}".format(self.selected_batch_id))
        try:
            current_reward = rewards[['batch_id' == self.selected_batch_id]]['afterBatchAuc']
            self.reward += current_reward
        except Exception:
            print(rewards.head())
            current_reward = -1
        info = {
            'batch_id': self.selected_batch_id,
            'iter_reward': current_reward
        }
        done = self.iteration > 20
        return obs, self.reward, done, info

    def reset(self):
        self.iteration = 0
        self.reward = 0
        self.selected_batch_id = -2

    def render(self, mode='human', close=False):
        print(self.reward)

    def meta_features_process(self, modelFiles, file_prefix, iteration, dataset):
        """
        :param modelFiles: folder of the meta-features
        :param file_prefix: the exp id and other file's prefix
        :param iteration:
        :param dataset: dataset name
        :return: the env, states and rewards based on the meta-features:
            env = scoreDist + dataset meta-features
            states = batch + instances meta-features
            rewards = AUC after add the batch
        """
        dataset_meta_features_folder = r'C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\CoMetDRL'

        '''states'''
        batch_meta_features = meta_model.loadBatchMetaData('{}\{}{}_Batches_Meta_Data.csv'
                                                           .format(modelFiles, file_prefix, iteration))
        instance_meta_features = meta_model.loadInstanceMetaData('{}\{}{}_Instances_Meta_Data.csv'
                                                                 .format(modelFiles, file_prefix, iteration))
        try:
            _state = pd.merge(batch_meta_features, instance_meta_features, how='left',
                              on=['exp_id', 'exp_iteration', 'batch_id'], left_index=True)
            del _state['afterBatchAuc']
        except Exception:
            print("fail to merge - state")

        '''env'''
        try:
            scoreDist_meta_features = meta_model.loadScoreDistMetaData('{}\{}{}_Score_Distribution_Meta_Data.csv'
                                                                       .format(modelFiles, file_prefix, iteration))
            scoreDist_meta_features['dataset'] = dataset
            dataset_meta_features = meta_model.loadDatasetMetaData(dataset, dataset_meta_features_folder)
            _env = pd.merge(scoreDist_meta_features, dataset_meta_features, how='left', on=['dataset'])
        except Exception:
            print("fail to merge - env")

        '''all meta-features'''
        meta_features = pd.merge(_state, _env, how='left',
                                 on=['exp_id', 'exp_iteration']
                                 , left_index=True)

        '''rewards'''
        _rewards = batch_meta_features[['batch_id', 'afterBatchAuc']]
        # _rewards = 0
        meta_features = meta_features.drop(['dataset'], axis=1)
        return _env, _state, _rewards, meta_features

    def set_df(self, new_df):
        self.df = new_df

    def get_iteration(self):
        return self.iteration
