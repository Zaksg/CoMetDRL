import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')


def get_exp_df(folder_prefix='all datasets'):
    all_folders = [f for f in listdir('./')]
    exp_df = pd.DataFrame(columns=['exp_id', 'dataset', 'test_file',
                                   'labeled_unlabeled_file'])
    for folder in all_folders:
        current_path = './{}'.format(folder)
        if folder.startswith(folder_prefix):
            files = [f for f in listdir(current_path) if isfile(join(current_path, f))]
            for file in files:
                if file.startswith('null'):
                    file = file[4:]
                if file.endswith('unlabeled_iteration_0.arff'):
                    file_list = file.split('_')
                    # file_prefix = '_'.join(file_list[:10])
                    file_prefix = '_'.join(file_list[:file_list.index('unlabeled')])
                    dataset = 'contraceptive' if file_list[0].startswith('unknow') else file_list[0]
                    exp_id = file_list[file_list.index('exp') + 1]
                    exp_df = exp_df.append({
                        'exp_id': exp_id,
                        'dataset': dataset,
                        'test_file': '{}/{}_testset.arff'.format(current_path, file_prefix),
                        'labeled_unlabeled_file': '{}/{}_'.format(current_path, file_prefix)
                        # 'unlabeled_iter_0': '{}/{}_unlabeled_iteration_0.arff'.format(current_path, file_prefix),
                        # 'unlabeled_iter_1': '{}/{}_unlabeled_iteration_1.arff'.format(current_path, file_prefix),
                        # 'labeled_iter_0': '{}/{}_labeled_iteration_0.arff'.format(current_path, file_prefix),
                        # 'labeled_iter_1': '{}/{}_labeled_iteration_1.arff'.format(current_path, file_prefix)
                    }, ignore_index=True)
    return exp_df


def bytes_to_string(dataset):
    str_df = dataset.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        dataset[col] = str_df[col]
    return dataset


def flatten_rows(dataset):
    for i, row in dataset.iterrows():
        res = ''
        for col_ind in range(dataset.shape[1] - 2):
            res = res + str(dataset.iloc[i, col_ind])
        if i == 0:
            res = res + str(dataset.iloc[i, col_ind+1])
        dataset.at[i, 'concat'] = res
    return dataset


def labeling_accuracy(exp_df, max_iterations=3):
    labeling_df = pd.DataFrame(columns=['exp_id', 'dataset', 'iteration',
                                   'total_added_labels_arff',
                                   'correct_added_labels_arff',
                                   'correct_labels_ratio_arff'])
    for iter in range(1, max_iterations):
        for ind, row in exp_df.iterrows():
            try:
                file_pattern = row['labeled_unlabeled_file']
                # labeled
                curr_labeled_data_arff = arff.loadarff('{}labeled_iteration_{}.arff'.format(file_pattern, iter))
                prev_labeled_data_arff = arff.loadarff('{}labeled_iteration_{}.arff'.format(file_pattern, (iter-1)))
                curr_labeled_data_df = pd.DataFrame(curr_labeled_data_arff[0])
                prev_labeled_data_df = pd.DataFrame(prev_labeled_data_arff[0])
                curr_added_labels = curr_labeled_data_df[(prev_labeled_data_df.shape[0]):]
                curr_added_labels = bytes_to_string(curr_added_labels).reset_index().drop(['index'], axis=1)
                curr_added_labels = flatten_rows(curr_added_labels)

                # unlabeled
                unlabeled_data_arff = arff.loadarff('{}unlabeled_iteration_{}.arff'.format(file_pattern, (iter-1)))
                unlabeled_data_prebious_iteration = pd.DataFrame(unlabeled_data_arff[0])
                udpi = bytes_to_string(unlabeled_data_prebious_iteration).reset_index().drop(['index'], axis=1)
                udpi = flatten_rows(udpi)

                # analysis
                count_correct, count_not_found, count_false = 0, 0, 0
                for i, row_2 in curr_added_labels.iterrows():
                    true_label = udpi[udpi['concat'] == row_2['concat']]
                    if len(true_label) >= 1:
                        true_label = true_label.iloc[0, -2]
                        pred_label = row_2.iloc[-2]
                        if true_label == pred_label:
                            count_correct = count_correct + 1
                        else:
                            count_false = count_false + 1
                    else:
                        count_not_found = count_not_found + 1
                total = count_correct + count_not_found + count_false
                labeling_df = labeling_df.append({
                    'exp_id': row['exp_id'],
                    'dataset': row['dataset'],
                    'iteration': iter,
                    'total_added_labels_arff': total,
                    'correct_added_labels_arff': count_correct,
                    'correct_labels_ratio_arff': count_correct / total
                }, ignore_index=True)

            except:
                continue
    return labeling_df


# def labeling_accuracy(dataset, exp_id, tested_iteration):
def labeling_accuracy_manually(exp_df):
    for ind, row in exp_df.iterrows():
        try:
            # labeled
            curr_labeled_data_arff = arff.loadarff(row['labeled_iter_1'])
            prev_labeled_data_arff = arff.loadarff(row['labeled_iter_0'])
            curr_labeled_data_df = pd.DataFrame(curr_labeled_data_arff[0])
            prev_labeled_data_df = pd.DataFrame(prev_labeled_data_arff[0])
            curr_added_labels = curr_labeled_data_df[(prev_labeled_data_df.shape[0]):]
            curr_added_labels = bytes_to_string(curr_added_labels).reset_index().drop(['index'], axis=1)
            curr_added_labels = flatten_rows(curr_added_labels)

            # unlabeled
            unlabeled_data_arff = arff.loadarff('./' + row['unlabeled_iter_0'])
            unlabeled_data_prebious_iteration = pd.DataFrame(unlabeled_data_arff[0])
            udpi = bytes_to_string(unlabeled_data_prebious_iteration).reset_index().drop(['index'], axis=1)
            udpi = flatten_rows(udpi)

            # analysis
            count_correct, count_not_found, count_false = 0, 0, 0
            for i, row_2 in curr_added_labels.iterrows():
                true_label = udpi[udpi['concat'] == row_2['concat']]
                if len(true_label) >= 1:
                    true_label = true_label.iloc[0, -2]
                    pred_label = row_2.iloc[-2]
                    if true_label == pred_label:
                        count_correct = count_correct + 1
                    else:
                        count_false = count_false + 1
                else:
                    count_not_found = count_not_found + 1
            total = count_correct + count_not_found + count_false
            exp_df.loc[ind, 'total_added_labels_arff'] = total
            exp_df.loc[ind, 'correct_added_labels_arff'] = count_correct
            exp_df.loc[ind, 'correct_labels_ratio_arff'] = count_correct / total
        except:
            continue
    return exp_df


def get_labeled_unlabeled_sets(labeled_arff, unlabeled_arff, is_drop_unlabeled_target=True):
    """
    Returns the labeled (train) and unlabeled (test) sets of a specific exp, iteration.
    The data of the unlabeled contains the target. By default we drop it.
    """
    # labeled
    labeled_data_arff = arff.loadarff(labeled_arff)
    labeled_data = pd.DataFrame(labeled_data_arff[0])
    # unlabeled
    unlabeled_data_arff = arff.loadarff(unlabeled_arff)
    unlabeled_data = pd.DataFrame(unlabeled_data_arff[0])
    if is_drop_unlabeled_target:
        unlabeled_data = unlabeled_data.iloc[:, :-1]
    # union dataset
    union_data = pd.concat([labeled_data, unlabeled_data])
    return labeled_data, unlabeled_data, union_data


if __name__ == "__main__":
    exp_arff = get_exp_df('model_file')
    # exp_arff = get_exp_df('dataset_objects')
    print("Dataframe created")

    # results = analyze_labels_tbl_based(exp_arff)
    max_iteration = 10
    results = labeling_accuracy(exp_arff, max_iteration)
    print("Labels analyzed")

    results.drop_duplicates(inplace=True)
    results.reset_index(inplace=True)
    results.to_csv('labeling_accuracy_arff.csv')
