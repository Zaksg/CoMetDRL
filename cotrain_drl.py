import pandas as pd
import numpy as np
import time
import sys
import traceback
import random
import subprocess
import meta_model_for_java_csv_v2 as meta_model
from os import listdir
from os.path import isfile, join
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as d_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN
import gym

from gym_comet_pck.gym_comet.envs.CoMetEnv import CoMetEnv

# import trfl
# import tensorflow as tf
# import stable_baselines


general_folder = '/data/home/zaksg/co-train/cotrain-v2/'


def meta_features_process(modelFiles, file_prefix, iteration, dataset):
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


def set_test_auc(dataset_arff, modelFiles, file_prefix):
    """

    :param dataset_arff: dataset arff file
    :param modelFiles: folder of the meta-features
    :param file_prefix: the exp id and other file's prefix
    :return: write auc scores to a csv file
    """
    auc_df = pd.DataFrame()
    auc_files = [f for f in listdir(modelFiles) if isfile(join(modelFiles, f))
                 and '_AUC_measures' in f and file_prefix in f]
    for auc_f in auc_files:
        temp_auc = pd.read_csv('{}{}'.format(modelFiles,auc_f))
        auc_df = pd.concat([auc_df, temp_auc], ignore_index=True)
    auc_df.to_csv('{}{}_exp_full_auc.csv'.format(modelFiles, file_prefix))


def drl_run():
    NUM_ITERATIONS = 20
    ''' Files '''
    # modelFiles = '/data/home/zaksg/co-train/cotrain-v2/model-files/'
    # modelFiles = '/Users/guyz/Documents/CoTrainingVerticalEnsemble - gilad/model files/'
    modelFiles = r"C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\meta_model\model_file_testing"

    '''step 1: init'''
    dataset_arff = "german_credit.arff"
    exp_id = int(round(time.time() % 1000000, 0))
    file_prefix = str(exp_id) + "_" + dataset_arff[:-5] + "_"
    subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "init", dataset_arff, file_prefix])
    subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration",
                     file_prefix, str(-2), str(0), str(exp_id)])
    score_ds_f, instance_batch_f, rewards, meta_f = meta_features_process(modelFiles, file_prefix, 0, dataset_arff)

    ''' RL '''
    env = DummyVecEnv([lambda: CoMetEnv(meta_f, dataset_arff, exp_id, modelFiles)])

    # model = PPO2(MlpPolicy, env, verbose=1)
    model = DQN(d_MlpPolicy, env, verbose=1, batch_size=1, exploration_fraction=0.5)
    model.learn(total_timesteps=NUM_ITERATIONS)

    obs = env.reset()
    for i in range(1, NUM_ITERATIONS):
        action, _states = model.predict(obs)
        # print("action: {}".format(action))
        # obs, rewards, done, info = env.step(action)
        obs, rewards, done, info = env.step(action)
        env.render()


def run_cotrain_iterations():
    '''
            Step 1: send java code the dataset (arff file) and file prefix
            Step 2: for 20 iteration - process the meta features and select the batch to add
                The java code provides objects of the dataset, the environment, the possible steps and the reward
        '''
    NUM_ITERATIONS = 20
    EPSILON = 0.05
    MAX_CANDIDATES = 1296
    # modelFiles = '/data/home/zaksg/co-train/cotrain-v2/model-files/'
    # modelFiles = '/Users/guyz/Documents/CoTrainingVerticalEnsemble - gilad/model files/'
    modelFiles = r"C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\meta_model\model_file_testing"

    '''step 1: init'''
    dataset_arff = "german_credit.arff"
    exp_id = int(round(time.time() % 1000000, 0))
    file_prefix = str(exp_id) + "_" + dataset_arff[:-5] + "_"
    subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "init", dataset_arff, file_prefix])

    '''step 2: learning'''
    for iteration in range(NUM_ITERATIONS):
        '''first iteration - no batch selection'''
        selected_batch_id = -2
        if iteration > 0:
            '''select batch id by epsilon greedy'''
            if random.random() <= EPSILON:
                selected_batch_id = random.randint(0, MAX_CANDIDATES - 1)
                # ToDo: calculate the reward
            else:
                # ToDo: implement selection by a RL method
                selected_batch_id = 1

        '''change env and get possible steps'''
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration",
                         file_prefix, str(selected_batch_id), str(iteration), str(exp_id)])

        '''Process meta features'''
        env, states, rewards, meta_features = meta_features_process(modelFiles, file_prefix, iteration, dataset_arff)
        print("Iteration: {}, meta features shape: {}X{}".format(iteration, str(meta_features.shape[0]), str(meta_features.shape[1])))
        print('Finish iteration {} on dataset {}'.format(iteration, dataset_arff))

    '''step 3: co-train evaluation'''
    set_test_auc(dataset_arff, modelFiles, file_prefix)


if __name__ == "__main__":
    drl_run()
    # run_cotrain_iterations()
