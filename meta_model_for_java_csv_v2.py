import pandas as pd
import time
import sys
import xgboost as xgb
import pickle
import traceback

general_folder = '/data/home/zaksg/co-train/cotrain-v2/'
'''Batch meta-data: load and pivot'''


def loadBatchMetaData(batch_meta_path):
    try:
        batch_meta_data_pivot_index = ['exp_id', 'exp_iteration', 'batch_id']
        df_instance_data_new_columns = [
            'batchDistanceFromAvgPartition_0_ins_1'
            , 'batchDistanceFromAvgPartition_0_ins_2'
            , 'batchDistanceFromAvgPartition_0_ins_3'
            , 'batchDistanceFromAvgPartition_0_ins_4'
            , 'batchDistanceFromAvgPartition_0_ins_5'
            , 'batchDistanceFromAvgPartition_0_ins_6'
            , 'batchDistanceFromAvgPartition_0_ins_7'
            , 'batchDistanceFromAvgPartition_0_ins_8'
            , 'batchDistanceFromAvgPartition_1_ins_1'
            , 'batchDistanceFromAvgPartition_1_ins_2'
            , 'batchDistanceFromAvgPartition_1_ins_3'
            , 'batchDistanceFromAvgPartition_1_ins_4'
            , 'batchDistanceFromAvgPartition_1_ins_5'
            , 'batchDistanceFromAvgPartition_1_ins_6'
            , 'batchDistanceFromAvgPartition_1_ins_7'
            , 'batchDistanceFromAvgPartition_1_ins_8']

        def loadDataAndPivot(filePath,
                             colNames,
                             pivotIndex,
                             colNames_pivot_instance,
                             isSaveCSV=False):
            batches_meta_data_full_df = pd.read_csv(filePath)

            '''get only the features with instance pos number'''
            data_instance = batches_meta_data_full_df[batches_meta_data_full_df['meta_feature_name'].apply(
                lambda x: x.startswith('batchDistanceBatchAucDifferenceFromAvgPartition_'))]
            data_instance['instance_number'] = data_instance.groupby(['exp_id', 'exp_iteration', 'batch_id'])[
                'att_id'].rank(ascending=False)
            '''get all the features without instance pos number'''
            data_no_instance = batches_meta_data_full_df[batches_meta_data_full_df['meta_feature_name'].apply(
                lambda x: not x.startswith('batchDistanceFromAvgPartition_'))]

            batches_meta_data_full_pivot_no_ins = pd.pivot_table(data_no_instance
                                                                 , values='meta_feature_value'
                                                                 , index=pivotIndex
                                                                 , columns=['meta_feature_name'])

            batches_meta_data_full_pivot_ins = pd.pivot_table(data_instance
                                                              , values='meta_feature_value'
                                                              , index=pivotIndex
                                                              , columns=['instance_number'])
            batches_meta_data_full_pivot_ins.columns = colNames_pivot_instance

            batches_meta_data_full_pivot = pd.concat(
                [batches_meta_data_full_pivot_no_ins, batches_meta_data_full_pivot_ins]
                , axis=1, join_axes=[batches_meta_data_full_pivot_no_ins.index])

            batches_meta_data_full_pivot = batches_meta_data_full_pivot.reset_index()
            return batches_meta_data_full_df, batches_meta_data_full_pivot

        batches_meta_data_full_df, batches_meta_data_final = loadDataAndPivot(
            filePath=batch_meta_path, colNames=''
            , pivotIndex=batch_meta_data_pivot_index
            , colNames_pivot_instance=df_instance_data_new_columns, isSaveCSV=False)

    except Exception as e:
        print('failed to batch meta features')
        print(traceback.format_exc())
        print(str(e))
        '''
        with open('{}_batch_meta_features_bugs.txt'.format(general_folder), 'a') as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        '''

    batches_meta_data_final = batches_meta_data_final.dropna(thresh=40)
    batches_meta_data_final = batches_meta_data_final.fillna(-1.0)
    return batches_meta_data_final


'''ScoreDist meta-data: load and pivot'''


def loadScoreDistMetaData(scoredist_meta_path):
    try:
        filePath = scoredist_meta_path
        # score_dist_meta_data_colNames = ['att_id', 'exp_iteration', 'exp_id', 'batch_id', 'meta_f_name','meta_f_val']

        # note: exp_id==iteration, exp_iteration==iteration id
        score_dist_pivotIndex = ['exp_id', 'exp_iteration']

        def loadDataAndPivot(filePath, colNames, pivotIndex, isSaveCSV, isFixNeeded):
            data = pd.read_csv(filePath)
            data = data.drop(['att_id', 'inner_iteration_id'], 1)
            data = data.fillna(-1.0)
            if isFixNeeded:
                features_to_fix_th = ['0.5001', '0.75', '0.9', '0.95']
                locs_to_change_5001 = [368, 369, 462, 463, 86, 87, 180, 181, 274, 275]
                locs_to_change_75 = [370, 371, 464, 465, 88, 89, 182, 183, 276, 277]
                locs_to_change_9 = [372, 373, 466, 467, 90, 91, 184, 185, 278, 279]
                locs_to_change_95 = [374, 375, 468, 469, 92, 93, 186, 187, 280, 281]
                for loc in locs_to_change_5001:
                    data.iloc[loc, data.columns.get_loc('meta_feature_name')] = data.iloc[loc, data.columns.get_loc(
                        'meta_feature_name')] + '_' + '0.5001'
                for loc in locs_to_change_75:
                    data.iloc[loc, data.columns.get_loc('meta_feature_name')] = data.iloc[loc, data.columns.get_loc(
                        'meta_feature_name')] + '_' + '0.75'
                for loc in locs_to_change_9:
                    data.iloc[loc, data.columns.get_loc('meta_feature_name')] = data.iloc[loc, data.columns.get_loc(
                        'meta_feature_name')] + '_' + '0.9'
                for loc in locs_to_change_95:
                    data.iloc[loc, data.columns.get_loc('meta_feature_name')] = data.iloc[loc, data.columns.get_loc(
                        'meta_feature_name')] + '_' + '0.95'

            data_pivot = pd.pivot_table(data, values='meta_feature_value', index=pivotIndex,
                                        columns=['meta_feature_name'])
            data_pivot = data_pivot.fillna(-1.0)

            data_pivot = data_pivot.reset_index()
            return data, data_pivot

        # colNames = #score_dist_meta_data_colNames,
        score_dist_df, score_dist_pivot = loadDataAndPivot(filePath, '',
                                                           score_dist_pivotIndex,
                                                           isSaveCSV=False,
                                                           isFixNeeded=True)
    except Exception as e:
        print('failed to insert score dist meta features')

    score_dist_pivot = score_dist_pivot.fillna(-1.0)
    return score_dist_pivot


'''Instance meta-data: load and pivot'''


def loadInstanceMetaData(instance_meta_path):
    try:
        filePath = instance_meta_path
        instance_pivot_index = ['exp_id', 'inner_iteration_id', 'exp_iteration', 'instance_pos', 'batch_id']
        instance_full_df = pd.read_csv(filePath)
        instance_full_df.fillna(-1.0)
        instance_full_df['meta_feature_value'] = pd.to_numeric(instance_full_df['meta_feature_value'], errors='coerce')
        instance_generic_features_pivot = pd.pivot_table(instance_full_df
                                                         , values='meta_feature_value'
                                                         , index=instance_pivot_index
                                                         , columns=['meta_feature_name'])
        instance_generic_features_pivot = instance_generic_features_pivot.fillna(-1.0)
        instance_generic_features_pivot = instance_generic_features_pivot.reset_index()
    except Exception as e:
        print('failed to insert instances meta data')
        traceback.print_exc()
    instance_generic_features_pivot = instance_generic_features_pivot.fillna(-1.0)

    '''prepare data for pivot by batch (union data by instances of a batch)'''
    instance_pivots_merged = instance_generic_features_pivot
    # instance_pivots_merged = instance_pivots_merged.reset_index(level='instance_pos')
    instance_pivots_merged['feature_number'] = \
        instance_pivots_merged.groupby(['exp_id', 'exp_iteration', 'batch_id'])['instance_pos'].cumcount()
    instance_pivots_merged = instance_pivots_merged.fillna(-1.0)
    '''group table by batch (concat instances)'''
    batch_instance_pivot = instance_pivots_merged[instance_pivots_merged['feature_number'] == 0]
    for f_number in range(1, 8):
        batch_instance_pivot = batch_instance_pivot.merge(
            instance_pivots_merged[instance_pivots_merged['feature_number'] == f_number]
            , on=['exp_id', 'exp_iteration', 'batch_id'], left_index=True, suffixes=('', '_' + str(f_number)))
    batch_instance_pivot = batch_instance_pivot.drop(['instance_pos', 'feature_number'
                                                         , 'feature_number_1', 'feature_number_2'
                                                         , 'feature_number_3', 'feature_number_4'
                                                         , 'feature_number_5', 'feature_number_6',
                                                      'feature_number_7'], axis=1)
    batch_instance_pivot = batch_instance_pivot.reset_index()
    return batch_instance_pivot


def loadDatasetMetaData(dataset_name, folder='/data/home/zaksg/co-train/cotrain-v2/meta-features'):
    dataset_meta_features = pd.read_csv('{}/dataset_meta_features.csv'.format(folder))
    dataset_meta_features = dataset_meta_features[dataset_meta_features['dataset'] == dataset_name]
    return dataset_meta_features


'''Read all meta-data tables and union'''


def createMetaFeatureTables(dataset_name, instance_meta_path, batch_meta_path, scoredist_meta_path):
    batch_meta_features = loadBatchMetaData(batch_meta_path)
    scoreDist_meta_features = loadScoreDistMetaData(scoredist_meta_path)
    instance_meta_features = loadInstanceMetaData(instance_meta_path)
    dataset_meta_features = loadDatasetMetaData(dataset_name)

    '''Union tables'''
    try:
        batch_meta_features['dataset'] = dataset_name
        second_merge = pd.merge(batch_meta_features, dataset_meta_features, how='left', on=['dataset'])
        # print("second merge complete")
    except Exception:
        print("fail to merge - second")
    try:
        third_merge = pd.merge(second_merge, scoreDist_meta_features, how='left', left_on=['exp_id', 'exp_iteration'],
                               right_on=['exp_iteration', 'exp_id'])
        third_merge = third_merge.rename({'exp_id_x': 'exp_id', 'exp_iteration_x': 'exp_iteration'}, axis=1)
        third_merge = third_merge.drop(columns=['exp_id_y', 'exp_iteration_y'])
        # print("third merge complete")
    except Exception:
        print("fail to merge - third")
    try:
        forth_merge = pd.merge(third_merge, instance_meta_features, how='left',
                               on=['exp_id', 'exp_iteration', 'batch_id']
                               , left_index=True)
        # print("forth merge complete")
    except Exception:
        print("fail to merge - forth")
    forth_merge = forth_merge.drop(['dataset'], axis=1).fillna(-1.0)
    return forth_merge


def fixClassificationValues(meta_features_current_iteration, dataset, folder_path):
    model_features_names = pd.read_csv('{}/feature_importance_{}.csv'.format(folder_path, dataset))
    model_features_names = model_features_names.iloc[:, 0].values
    current_iteration_features_names = meta_features_current_iteration.columns
    feature_list = list(set().union(current_iteration_features_names, model_features_names))

    for feature in model_features_names:
        if feature not in current_iteration_features_names:
            meta_features_current_iteration[feature] = -1.0
    for feature in current_iteration_features_names:
        if feature not in model_features_names:
            meta_features_current_iteration = meta_features_current_iteration.drop([feature], axis=1)
    meta_features_current_iteration = meta_features_current_iteration.loc[:,
                                      ~meta_features_current_iteration.columns.duplicated()]
    return meta_features_current_iteration


'''Rank'''


def getDatasetName(abs_path):
    input_file_directory = '/data/home/zaksg/co-train/inputData/'
    dataset_name = abs_path.replace(input_file_directory, '')
    return dataset_name


def loadRankingModel(dataset_name, folder_path):
    model = xgb.XGBRanker(objective='rank:map', learning_rate=0.1, gamma=1.0, max_depth=6, n_estimators=4)
    model.load_model('{}/{}_xgboost_rank_model.bin'.format(folder_path, dataset_name))
    return model


def loadClassifictaionModel(dataset_name, folder_path):
    filename = '{}/rfc_model_{}.sav'.format(folder_path, dataset_name)
    return pickle.load(open(filename, 'rb'))


def runLoadedModel(meta_features_current_iteration, model, model_type):
    if model_type == 'ranking':
        pred = model.predict(meta_features_current_iteration)
    else:
        meta_features_current_iteration = meta_features_current_iteration.fillna(-1.0)
        pred = model.predict_proba(meta_features_current_iteration)[:, 1]
    ranked_batch = meta_features_current_iteration[['exp_id', 'exp_iteration', 'batch_id']]
    ranked_batch['rank'] = pred
    max_score = pred.max()
    millis = int(round(time.time() * 1000))
    top_ranked_batches = ranked_batch[ranked_batch['rank'] == max_score]
    # top_ranked_batches.to_csv(
    #     'C:/Users/guyz/Documents/CoTrainingVerticalEnsemble/meta_model/meta model sorted classification/top_ranked_{}.csv'
    #         .format(millis))
    if -1 in top_ranked_batches[['batch_id']].values:
        selected_batch = top_ranked_batches[top_ranked_batches['batch_id'] == -1]
    else:
        selected_batch = top_ranked_batches.sample(1)
    return selected_batch


if __name__ == "__main__":
    org_dist_list = [
        "ailerons.arff"
        , "bank-full.arff"
        , "cardiography_new.arff"
        , "contraceptive.arff"
        , "cpu_act.arff"
        , "delta_elevators.arff"
        , "diabetes.arff"
        , "german_credit.arff"
        , "ionosphere.arff"
        , "kc2.arff"
        , "mammography.arff"
        , "page-blocks_new.arff"
        , "php0iVrYT.arff"
        , "php7KLval.arff"
        , "php8Mz7BG.arff"
        , "php9xWOpn.arff"
        , "php50jXam.arff"
        , "phpelnJ6y.arff"
        , "phpOJxGL9.arff"
        , "puma8NH.arff"
        , "puma32H.arff"
        , "seismic-bumps.arff"
        , "space_ga.arff"
        , "spambase.arff"
        , "wind.arff"]  # didn't work: "kc2.arff"
    try:
        model = 'classification'
        ''' test files '''
        test_folder = r"C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\CoMetDRL\model_file_testing"
        dataset_name = "german_credit.arff"
        instance_meta_path = r"{}\171130_german_credit_0_Instances_Meta_Data.csv".format(test_folder)
        batch_meta_path = r"{}\171130_german_credit_0_Batches_Meta_Data.csv".format(test_folder)
        scoredist_meta_path = r"{}\171130_german_credit_0_Score_Distribution_Meta_Data.csv".format(test_folder)

        '''
        dataset_name = getDatasetName(sys.argv[1])
        instance_meta_path = sys.argv[2]
        batch_meta_path = sys.argv[3]
        scoredist_meta_path = sys.argv[4]
        '''

        if dataset_name not in org_dist_list:
            dataset_name = "space_ga.arff"
        meta_features_current_iteration = createMetaFeatureTables(dataset_name, instance_meta_path, batch_meta_path,
                                                                  scoredist_meta_path)
        if model == 'ranking':
            models_folder_path = '/data/home/zaksg/co-train/cotrain-v2/meta-features'
            # models_folder_path = 'C:/Users/guyz/Documents/CoTrainingVerticalEnsemble/meta_model/meta model qa'
            loaded_model = loadRankingModel(dataset_name, models_folder_path)
            meta_features_current_iteration = fixClassificationValues(meta_features_current_iteration
                                                                      , dataset_name, models_folder_path)
        else:
            folder_path = 'C:/Users/guyz/Documents/CoTrainingVerticalEnsemble/meta_model'
            # folder_path = '/data/home/zaksg/co-train/cotrain-v2/meta-features'
            loaded_model = loadClassifictaionModel(dataset_name, folder_path)
            meta_features_current_iteration = fixClassificationValues(meta_features_current_iteration
                                                                      , dataset_name, folder_path)
        selected_batch = runLoadedModel(meta_features_current_iteration, loaded_model, model)
        print(selected_batch['batch_id'].values[0])
    except Exception as e:
        print(sys.exc_info())
        print(e)
