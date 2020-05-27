import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import meta_model_for_java_csv_v2 as meta_model
import subprocess
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_AUC_SCORE = 10000
ITERATIONS_PER_DATASET = 10
# BATCH_CANDIDATES = 1296
BATCH_CANDIDATES = 625
EPSILON = 0.05
RUNS_PER_DATASET = 5

# consider: in subprocess.call: shell = True

class CoMetEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset_env_list, exp_id, modelFiles):
        super(CoMetEnv, self).__init__()


        ''' Env attributes '''
        self.cnt_dataset_runs = 0
        self.num_datasets = len(dataset_env_list)
        self.dataset_list = dataset_env_list
        self.current_dataset_index = 0
        self.exp = exp_id
        self.dataset = self.dataset_list[self.current_dataset_index]
        self.modelFiles = modelFiles
        self.iteration = 0
        self.file_prefix = str(self.exp) + "_" + self.dataset[:-5] + "_"
        # self.selected_batch_id = -2
        ''' Env creation '''
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "init",
                         self.dataset, self.file_prefix, str(self.exp)])
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration",
                         self.file_prefix, str(-2), str(self.iteration), str(self.exp)])
        score_ds_f, instance_batch_f, rewards, meta_f = self.meta_features_process(modelFiles, self.file_prefix,
                                                                                   self.iteration, self.dataset)
        self.df = meta_f
        self.df_col_list = meta_f.columns

        ''' Reward attributes '''
        self.reward = 0
        self.prev_auc = 0
        self.cost = 0
        ''' GYM attributes '''
        # Reward range
        self.reward_range = (0, MAX_AUC_SCORE * ITERATIONS_PER_DATASET * RUNS_PER_DATASET)

        # Actions
        self.action_space = spaces.Discrete(BATCH_CANDIDATES)
        # self.action_space = spaces.MultiDiscrete(nvec=[BATCH_CANDIDATES, 2]) # For active learning

        # 625 / 1296 batches and 2232 meta-features per batch
        high = np.inf
        low = -high
        # self.observation_space = spaces.Box(low=-exp_id, high=exp_id, shape=(BATCH_CANDIDATES, len(self.df_col_list)), dtype=np.float16)
        self.observation_space = spaces.Box(low=low, high=high,
            shape=(BATCH_CANDIDATES, len(self.df_col_list)), dtype=np.float16)

    def _next_observation(self):
        print("iteration: {}, selected batch: {}".format(self.iteration, self.selected_batch_id))
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration",
                         self.file_prefix, str(self.selected_batch_id), str(self.iteration), str(self.exp)])
        score_ds_f, instance_batch_f, rewards, meta_f = self.meta_features_process(
            self.modelFiles, self.file_prefix, self.iteration, self.dataset)
        not_in_meta_f = list(set(self.df_col_list) - set(meta_f.columns))
        more_cols = {}
        for col in not_in_meta_f:
            if col not in more_cols:
                more_cols[col] = -1.0
        meta_f = meta_f.assign(**more_cols)
        meta_f = meta_f[self.df_col_list]
        meta_f = meta_f.loc[:, ~meta_f.columns.duplicated()]
        self.df = meta_f
        return meta_f, rewards

    def _take_action(self, action, act_type):
        print("action - take action: {}".format(action))
        '''
        if random.random() <= EPSILON:
            print("Entered the epsilon")
            self.selected_batch_id = random.randint(0, BATCH_CANDIDATES - 1)
        else:
            # self.selected_batch_id = action
            self.selected_batch_id = int(self.df.iloc[action]['batch_id'])
        '''
        self.selected_batch_id = int(self.df.iloc[action]['batch_id'])
        if act_type == 1:
            self.cost += 1
            print("Use active learning")
        elif act_type == 0:
            print("Don't use active learning")
        else:
            print("Multi discrete not activated")

    def step(self, action):
        self.iteration += 1
        try:
            act = action[0]
            act_type = action[1]
        except Exception:
            act = action
            act_type = -1
        self._take_action(act, act_type)
        obs, rewards = self._next_observation()
        current_auc = MAX_AUC_SCORE * rewards[rewards['batch_id'] == self.selected_batch_id]['afterBatchAuc'].values[0]
        if self.iteration <= 1:
            # self.reward = current_auc
            self.prev_auc = current_auc
        else:
            self.reward += (current_auc - self.prev_auc)
            print("reward: {}, current auc: {}, prev auc: {}".format(self.reward, current_auc, self.prev_auc))
            self.prev_auc = current_auc
        info = {
            'batch_id': self.selected_batch_id,
            'iter_auc': current_auc,
            'reward': self.reward
        }
        ''' Datasets changes '''
        done = False
        if self.iteration >= ITERATIONS_PER_DATASET:
            # no more iterations and do not need to run same ds
            if self.cnt_dataset_runs >= RUNS_PER_DATASET:
                # ds left to run
                if self.current_dataset_index < self.num_datasets - 1:
                    self.cnt_dataset_runs = 0
                    self.current_dataset_index += 1
                    self.dataset = self.dataset_list[self.current_dataset_index]
                    print("Start new DS {}".format(self.dataset))
                    self.run_dataset_new_seed(self.dataset)
                # out of exp
                else:
                    print("Done")
                    done = True
            # no more iterations, need to run same ds
            else:
                print("Continue to another run of {}".format(self.dataset))
                self.run_dataset_new_seed(self.dataset)
                self.cnt_dataset_runs += 1
        print("DS: {}. Iteration {} ".format(self.dataset, self.iteration))

        return obs, self.reward, done, info

    def reset(self):
        if self.iteration >= ITERATIONS_PER_DATASET:
            self.exp += 1
            # self.selected_batch_id = -2
            subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "init", self.dataset, self.file_prefix,
                             str(self.exp)])
            subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration",
                             self.file_prefix, str(-2), str(0), str(self.exp)])
            # score_ds_f, instance_batch_f, rewards, meta_f = self.meta_features_process(self.modelFiles, self.file_prefix, 0, self.dataset)
            self.iteration = 0
            self.reward = 0
            self.cost = 0
            self.prev_auc = 0
            self.selected_batch_id = random.randint(0, BATCH_CANDIDATES - 1)
        else:
            print("Try to reset before max iteration. Currently: {}; Should be {}".format(self.iteration, ITERATIONS_PER_DATASET))

    def render(self, mode='human', close=False):
        print("Selected batch: {}".format(self.selected_batch_id))
        print("Reward: {}".format(self.reward))

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
        # dataset_meta_features_folder = r'C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\CoMetDRL'
        dataset_meta_features_folder = modelFiles

        '''states'''
        batch_meta_features = meta_model.loadBatchMetaData('{}/{}{}_Batches_Meta_Data.csv'
                                                           .format(modelFiles, file_prefix, iteration))
        instance_meta_features = meta_model.loadInstanceMetaData('{}/{}{}_Instances_Meta_Data.csv'
                                                                 .format(modelFiles, file_prefix, iteration))
        try:
            _state = pd.merge(batch_meta_features, instance_meta_features, how='left',
                              on=['exp_id', 'exp_iteration', 'batch_id'], left_index=True)
            del _state['afterBatchAuc']
        except Exception:
            print("fail to merge - state")

        '''env'''
        try:
            scoreDist_meta_features = meta_model.loadScoreDistMetaData('{}/{}{}_Score_Distribution_Meta_Data.csv'
                                                                       .format(modelFiles, file_prefix, iteration))
            scoreDist_meta_features['dataset'] = dataset
            dataset_meta_features = meta_model.loadDatasetMetaData(dataset, dataset_meta_features_folder)
            _env = pd.merge(scoreDist_meta_features, dataset_meta_features, how='left', on=['dataset'])
        except Exception as e:
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

    def set_id(self, new_exp_id):
        self.exp = new_exp_id

    def run_dataset_new_seed(self, dataset_name):
        self.exp += 1
        self.file_prefix = str(self.exp) + "_" + dataset_name[:-5] + "_"
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "init", dataset_name, self.file_prefix,
                         str(self.exp)])
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration",
                         self.file_prefix, str(-2), str(0), str(self.exp)])
        score_ds_f, instance_batch_f, rewards, meta_f = self.meta_features_process(self.modelFiles, self.file_prefix, 0,
                                                                                   dataset_name)
        self.df = meta_f
        self.iteration = 0
        self.prev_auc = 0
        # self.selected_batch_id = random.randint(0, BATCH_CANDIDATES - 1)
