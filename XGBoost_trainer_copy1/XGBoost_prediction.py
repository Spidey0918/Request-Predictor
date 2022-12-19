import pandas as pd
import xgboost as xgb
import sys


def index_range_parser(range_string):
    ranges = [index.strip() for index in range_string.split(',')]
    expanded_range = []

    for index_range in ranges:
        indices = index_range.split('-')
        start_ind = int(indices[0])
        if len(indices) < 2:
            end_ind = start_ind
        else:
            end_ind = int(indices[1])
        expanded_range.extend(range(start_ind, end_ind + 1))
    return expanded_range


def xgboost_prediction(model_path, features):
    trained_model = pd.read_pickle(model_path)
    return trained_model.predict_proba(features)[:, 1]


if __name__ == '__main__':
    input_data = pd.read_csv(sys.argv[1], header=None, sep='\t')
    feature_indices = index_range_parser(sys.argv[3])
    keep_columns_indices = index_range_parser(sys.argv[5])
    if sys.argv[4] != '-':
        category_ind = index_range_parser(sys.argv[4])
        input_data.iloc[:, category_ind] = input_data.iloc[:, category_ind].astype('category')
    predictions = xgboost_prediction(sys.argv[2], input_data.iloc[:, feature_indices])
    print(predictions)
    output_data = input_data.iloc[:, keep_columns_indices].copy()
    output_data['Score'] = predictions
    output_data.to_csv(sys.argv[6], sep='\t', index=False)
