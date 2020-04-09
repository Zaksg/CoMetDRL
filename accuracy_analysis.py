import pandas as pd
import glob

general_folder = '/data/home/zaksg/co-train/cotrain-v2/drl-co'
modelFiles = "/data/home/zaksg/co-train/cotrain-v2/drl-co/model_files"
# modelFiles = r"C:\Users\guyz\Documents\CoTrainingVerticalEnsemble\meta_model\model_file_testing"

pred_analysis_files = glob.glob('{}/*.csv'.format(modelFiles))
auc_files = []
labeling_files = []

for filename in pred_analysis_files:
    if "AUC" in filename:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['file_name'] = filename[len(modelFiles):]
        df['ds'] = filename[len(modelFiles):].split('_')[1]
        auc_files.append(df)
    elif "Selected_Batch_Analysis" in filename:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['file_name'] = filename[len(modelFiles):]
        df['ds'] = filename[len(modelFiles):].split('_')[1]
        dfg = df.groupby(['exp_id', 'exp_iteration', 'batch_id', 'file_name', 'ds'])['is_correct'].sum()
        dfg = dfg.reset_index()
        labeling_files.append(dfg)

accuracy_df = pd.concat(auc_files, axis=0, ignore_index=True)
accuracy_df.to_csv('{}/auc_analysis.csv'.format(general_folder))

labeling_df = pd.concat(labeling_files, axis=0, ignore_index=True)
labeling_df.to_csv('{}/labeling_analysis.csv'.format(general_folder))
