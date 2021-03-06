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
import gym
from gym_comet_pck.gym_comet.envs.CoMetEnv import CoMetEnv
# import tensorflow as tf
import warnings
from stable_baselines.deepq.policies import MlpPolicy as d_MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

'''
## see keras-rl2 requirements for tensorflow version
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam
# from tf.keras.optimizers import Adam
from rl.agents import DQNAgent, SARSAAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.layers import Input, Embedding, Dense, Concatenate, Conv1D, Lambda, Reshape
from keras.models import Model
import keras.backend as K
# import trfl
'''
# general_folder = '/data/home/zaksg/co-train/cotrain-v2/'
general_folder = '/data/home/zaksg/co-train/cotrain-v2/drl-co'
warnings.simplefilter(action='ignore', category=FutureWarning)
DATASET_LIST_SUB = [
    'abalone.arff',
    'ailerons.arff',
    'contraceptive.arff',
    'fri_c2_1000_25.arff',
    'german_credit.arff',
    'ozone-level-8hr.arff',
    'puma8NH.arff',
    'seismic-bumps.arff',
    'space_ga.arff',
    'spambase.arff'
]

DATASET_LIST_FULL = [
    'abalone.arff',
    'ailerons.arff',
    'cardiography_new.arff',
    'contraceptive.arff',
    'cpu_act.arff',
    'delta_elevators.arff',
    'diabetes.arff',
    'fri_c0_1000_50.arff',
    'fri_c2_1000_10.arff',
    'fri_c2_1000_25.arff',
    'german_credit.arff',
    'ionosphere.arff',
    'kc2.arff',
    'kdd_JapaneseVowels.arff',
    'kr-vs-kp.arff',
    'mammography.arff',
    'mfeat-karhunen.arff',
    'ozone-level-8hr.arff',
    'page-blocks_new.arff',
    'php0iVrYT.arff',
    'php7KLval.arff',
    'php8Mz7BG.arff',
    'phpelnJ6y.arff',
    'phpOJxGL9.arff',
    'puma32H.arff',
    'puma8NH.arff',
    'qsar-biodeg.arff',
    'seismic-bumps.arff',
    'space_ga.arff',
    'spambase.arff',
    'wilt.arff',
    'wind.arff'
]

DATASET_GROUP_1 = [
    'abalone.arff',
    'ailerons.arff',
    'cpu_act.arff',
    'delta_elevators.arff',
    'kdd_JapaneseVowels.arff',
    'puma32H.arff',
    'space_ga.arff',
    'wind.arff'
]

DATASET_GROUP_2 = [
    'fri_c0_1000_50.arff',
    'ionosphere.arff',
    'kr-vs-kp.arff',
    'mfeat-karhunen.arff',
    'ozone-level-8hr.arff',
    'qsar-biodeg.arff',
    'spambase.arff',
    'phpelnJ6y.arff'
]

DATASET_GROUP_3 = [
    'php7KLval.arff',
    'php0iVrYT.arff',
    'contraceptive.arff',
    'diabetes.arff',
    'fri_c2_1000_10.arff',
    'fri_c2_1000_25.arff',
    'german_credit.arff',
    'phpOJxGL9.arff'
]

DATASET_GROUP_4 = [
    'kc2.arff',
    'puma8NH.arff',
    'seismic-bumps.arff',
    'cardiography_new.arff',
    'mammography.arff',
    'page-blocks_new.arff',
    'php8Mz7BG.arff',
    'wilt.arff'
]

problematic_ds = [
    'ionosphere.arff'
]

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
    # dataset_meta_features_folder = r'C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\CoMetDRL'
    dataset_meta_features_folder = modelFiles

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


def drl_run_sb(dataset_arff, modelFiles, model_name=None):
    NUM_ITERATIONS = 20
    '''step 1: init'''
    exp_id = int(round(time.time() % 1000000, 0))
    if not model_name:
        model_name='_'.join(dataset_arff).replace(".arff", "")

    ''' RL '''
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/{}'.format(model_name))
    event_callback = EveryNTimesteps(n_steps=40, callback=checkpoint_on_event)

    env = DummyVecEnv([lambda: CoMetEnv(dataset_arff, exp_id, modelFiles)])

    # model = PPO2(MlpPolicy, env, verbose=1, learning_rate=0.01)
    # model = A2C(MlpPolicy, env, verbose=1)
    model = DQN(d_MlpPolicy, env, verbose=1, batch_size=1, exploration_fraction=0.8)

    model.learn(total_timesteps=len(dataset_arff) * 12 * 5, callback=event_callback)
    model_name = "dqn_{}".format(model_name)
    # model_name = "ppo_{}".format(dataset_arff[:-5])

    model.save(model_name)
    env.close()
    return model_name


def drl_run_test_dataset(dataset, modelFiles, trained_model):
    NUM_ITERATIONS = len(dataset) * 5 * 11

    model = DQN.load(trained_model)
    # model = PPO2.load(trained_model)

    print("model loaded")
    '''step 1: init'''
    exp_id = int(round(time.time() % 1000000, 0))

    ''' RL '''
    env = DummyVecEnv([lambda: CoMetEnv(dataset, exp_id, modelFiles)])
    model.set_env(env)
    # model.learn(total_timesteps=NUM_ITERATIONS)
    obs = env.reset()
    done = False
    i = 0
    while not done and i < NUM_ITERATIONS:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        i += 1
        env.render()


def drl_run_keras(dataset_arff, modelFiles):
    pass
    '''
    ValueError: Error when checking input: expected dense_input to have 3 dimensions
        , but got array with shape (1, 1, 1, 1296, 2232)

    :param dataset_arff:
    :param modelFiles:
    :return:
    '''
    '''
    # Get the environment and extract the number of actions.
    exp_id = int(round(time.time() % 1000000, 0))
    env = DummyVecEnv([lambda: CoMetEnv(dataset_arff, exp_id, modelFiles)])
    # np.random.seed(123)
    # env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
    # model.add(tf.keras.layers.InputLayer(input_shape=env.observation_space.shape))
    model.add(tf.keras.layers.Dense(16, input_shape=env.observation_space.shape))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(nb_actions, activation='linear'))
    # model.add(tf.keras.layers.Reshape((nb_actions, 1)))
    # model.add(tf.keras.layers.Reshape((nb_actions,)))

    memory = SequentialMemory(limit=50000, window_length=1)
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy()

    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
    #                enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(tf.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    model_name = 'dqn_{}_weights.h5f'.format('_'.join(dataset_arff).replace(".arff", ""))
    dqn.save_weights(model_name, overwrite=True)
    '''
    '''
    #Sarsa
    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    sarsa.compile(tf.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])
    sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2)
    
    return model_name
    '''


def run_cotrain_iterations(dataset_arff="german_credit.arff"):
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
    # dataset_arff = "cardiography_new.arff"
    exp_id = int(round(time.time() % 1000000, 0))
    file_prefix = str(exp_id) + "_" + dataset_arff[:-5] + "_"
    subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "init", dataset_arff, file_prefix, str(exp_id)])

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
    """
        Args: None or [1] TYPE, [2] Dataset
    """
    # TYPE can be: train, test, full-run
    # TYPE = "test"
    TYPE = "train"

    ''' Files '''
    # modelFiles = r"C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\meta_model\model_file_testing"
    modelFiles = "/data/home/zaksg/co-train/cotrain-v2/drl-co/model_files"

    ''' Datasets '''

    ## train datasets
    # dataset_arff_train = ["german_credit.arff", "contraceptive.arff", "ionosphere.arff"]
    # dataset_arff_train = DATASET_LIST_SUB
    dataset_arff_train = DATASET_LIST_FULL

    ## test datasets
    test_dataset_list = []
    trained_model_name = 'dqn_0.8_exploration_sub_ds_list.zip'

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            print("Starting test")
            TYPE = "test"
        elif sys.argv[1] == "run":
            TYPE = "run"
        elif sys.argv[1] == "train":
            TYPE = "train"
        if sys.argv[2] == "sub_list":
            test_dataset_list = DATASET_LIST_SUB
        elif sys.argv[2] == "1":
            test_dataset_list = DATASET_GROUP_1
            trained_model_name = 'dqn_1.zip'
        elif sys.argv[2] == "2":
            test_dataset_list = DATASET_GROUP_2
            trained_model_name = 'dqn_2.zip'
        elif sys.argv[2] == "3":
            test_dataset_list = DATASET_GROUP_3
            trained_model_name = 'dqn_3.zip'
        elif sys.argv[2] == "4":
            test_dataset_list = DATASET_GROUP_4
            trained_model_name = 'dqn_4.zip'
        else:
            test_dataset_list.append(sys.argv[2])
    else:
        test_dataset_list.append("cardiography_new.arff")
    # index of test list
    if len(sys.argv) > 3:
        dataset_arff_test = [test_dataset_list[int(sys.argv[3])]]
    else:
        dataset_arff_test = test_dataset_list

    ''' Run '''
    if TYPE == "train":
        trained_model_name = drl_run_sb(list(set(dataset_arff_train) - set(test_dataset_list) - set(problematic_ds)),
                                        modelFiles, sys.argv[2])
    elif TYPE == "test":
        print("model: {}".format(trained_model_name))
        print("ds: {}".format(dataset_arff_test))
        # trained_model_name = 'dqn_0.6_exploration_sub_ds_list.zip'
        # trained_model_name = 'dqn_0.8_exploration_sub_ds_list.zip'
        drl_run_test_dataset(dataset_arff_test, modelFiles, trained_model_name)
    else:
        trained_model_name = drl_run_sb(dataset_arff_train, modelFiles)
        drl_run_test_dataset(dataset_arff_test, modelFiles, trained_model_name)


    ## Original co training + meta features extraction
    # run_cotrain_iterations(dataset_arff_train)

