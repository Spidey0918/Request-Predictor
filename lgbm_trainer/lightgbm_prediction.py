from string import digits
import numpy as np
import pandas as pd
import lightgbm as lgbm
from parsing_utility import index_range_parser
from sklearn.metrics import classification_report, confusion_matrix
import sys

headers = sys.argv[7].split(',')
threshold = eval(sys.argv[8])


def compute_metrics(df, label_col, pred_col):
    labels = df[label_col].tolist()
    pred_labels = df[pred_col].tolist()
    classification_metric = classification_report(
        labels, pred_labels, digits=4)
    confusion_mat = confusion_matrix(labels, pred_labels)

    print('Metrics Report: ')
    print(classification_metric)
    print('Confustion Matrix: ')
    print(confusion_mat)


def lightgbm_prediction(model_path, features):
    trained_model = lgbm.Booster(model_file=model_path)
    return trained_model.predict(features)


if __name__ == '__main__':
    input_data = pd.read_csv(sys.argv[1], names=headers, sep='\t')
    print("Test Data Size: ", input_data.shape[0])
    feature_indices = index_range_parser(sys.argv[3])
    keep_columns_indices = index_range_parser(sys.argv[5])
    if sys.argv[4] != '-':
        category_ind = index_range_parser(sys.argv[4])
        input_data.iloc[:, category_ind] = input_data.iloc[:,
                                                           category_ind].astype('category')
    predictions = lightgbm_prediction(
        sys.argv[2], input_data.iloc[:, feature_indices])
    output_data = input_data.iloc[:, keep_columns_indices].copy()
    output_data['Score'] = predictions
    output_data['IsShown_Pred'] = np.where(
        output_data['Score'] > threshold, 1, 0)
    compute_metrics(output_data, 'IsShown', 'IsShown_Pred')
    output_data.to_csv(sys.argv[6], sep='\t', index=False)
