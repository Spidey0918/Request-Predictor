# START: OWN CODE
import numpy as np
import pandas as pd
import shap
import sys
import lightgbm as lgb
from parsing_utility import index_range_parser
pd.options.display.float_format = '{:.10f}'.format


def get_feature_importance(data, shap_vals, topk=100):
    vals = np.abs(shap_vals).mean(0)
    feature_importance = pd.DataFrame(list(zip(data.columns, vals)), columns=[
                                      'feature_name', 'feature_importance_vals'])
    feature_importance.sort_values(
        by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance.head(topk)


if __name__ == "__main__":
    model = lgb.Booster(model_file=sys.argv[2])
    feature_indices = index_range_parser(sys.argv[3])
    category_indices = index_range_parser(sys.argv[4])
    features = sys.argv[5].split(',')
    print(features)
    input_data = pd.read_csv(sys.argv[1], names=features, sep='\t')
    if category_indices != '-':
        input_data.iloc[:, category_indices] = input_data.iloc[:,
                                                               category_indices].astype('category')
    input_data = input_data.iloc[:, feature_indices]
    input_data = input_data.sample(100000)
    model.params['objective'] = 'regression'
    exp = shap.TreeExplainer(model)
    shap_val = exp.shap_values(input_data)
    shap_importance = get_feature_importance(input_data, shap_val)
    shap_importance.to_csv(sys.argv[6], sep="\t", header=[
                           "Feature", "Shap Value"])
# END: OWN CODE
