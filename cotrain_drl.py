import pandas as pd
import time
import sys
import traceback
import random
import subprocess
import meta_model_for_java_csv_v2 as meta_model
from os import listdir
from os.path import isfile, join

# import trfl
# import tensorflow as tf
# import stable_baselines


general_folder = '/data/home/zaksg/co-train/cotrain-v2/'


'''
    Return the env, states and rewards based on the meta-features
    env = scoreDist + dataset meta-features
    states = batch + instances meta-features
    rewards = AUC difference (available in the batch meta-features) 
'''
def meta_features_process(modelFiles, file_prefix, iteration, dataset):
    # dataset_meta_features_folder = r'C:\Users\guyz\Documents\CoTrainingVerticalEnsemble - gilad\CoTrainingVerticalEnsemble\out\artifacts\CoTrainingVerticalEnsembleV2_2_jar'
    dataset_meta_features_folder = r'C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\CoMetDRL'

    '''states'''
    batch_meta_features = meta_model.loadBatchMetaData('{}\{}{}_Batches_Meta_Data.csv'
                                                       .format(modelFiles, file_prefix, iteration))
    instance_meta_features = meta_model.loadInstanceMetaData('{}\{}{}_Instances_Meta_Data.csv'
                                                             .format(modelFiles, file_prefix, iteration))
    try:
        _state = pd.merge(batch_meta_features, instance_meta_features, how='left',
                          on=['exp_id', 'exp_iteration', 'batch_id'], left_index=True)
        del _state['BatchAucDifference']
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

    '''rewards'''
    _rewards = batch_meta_features["afterBatchAuc"]
    # _rewards = 0

    return _env, _state, _rewards


def set_test_auc(dataset_arff, modelFiles, file_prefix):
    auc_df = pd.DataFrame()
    auc_files = [f for f in listdir(modelFiles) if isfile(join(modelFiles, f))
                 and '_AUC_measures' in f and file_prefix in f]
    for auc_f in auc_files:
        temp_auc = pd.read_csv('{}{}'.format(modelFiles,auc_f))
        auc_df = pd.concat([auc_df, temp_auc], ignore_index=True)
    auc_df.to_csv('{}{}_exp_full_auc.csv'.format(modelFiles, file_prefix))

if __name__ == "__main__":
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
        subprocess.call(['java', '-jar', 'CoTrainingVerticalEnsembleV2.jar', "iteration", file_prefix, str(selected_batch_id), str(iteration), str(exp_id)])

        '''Process meta features'''
        env, states, rewards = meta_features_process(modelFiles, file_prefix, iteration, dataset_arff)
        print('Finish iteration {} on dataset {}'.format(iteration, dataset_arff))

    '''step 3: co-train evaluation'''
    set_test_auc(dataset_arff, modelFiles, file_prefix)
