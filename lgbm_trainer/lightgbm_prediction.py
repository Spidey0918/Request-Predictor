# START: OWN CODE
import pandas as pd
import lightgbm as lgbm
from parsing_utility import index_range_parser
import sys


def lightgbm_prediction(model_path, features):
    trained_model = lgbm.Booster(model_file=model_path)
    return trained_model.predict(features)


if __name__ == '__main__':
    input_data = pd.read_csv(sys.argv[1], header=None, sep='\t')
    feature_indices = index_range_parser(sys.argv[3])
    keep_columns_indices = index_range_parser(sys.argv[5])
    if sys.argv[4] != '-':
        category_ind = index_range_parser(sys.argv[4])
        input_data.iloc[:,category_ind] = input_data.iloc[:,category_ind].astype('category')
    predictions = lightgbm_prediction(sys.argv[2], input_data.iloc[:, feature_indices])
    output_data = input_data.iloc[:, keep_columns_indices].copy()
    output_data['Score'] = predictions
    output_data.to_csv(sys.argv[6], sep='\t', index=False)
# END: OWN CODE
