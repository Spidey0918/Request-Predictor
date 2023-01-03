import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
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


if __name__ == '__main__':
    # model_file = open(sys.argv[1], 'r', encoding='ISO-8859-1')
    # model_info = model_file.read()
    # model_info = json_normalize(model_info)
    # model = pd.DataFrame(model_info)
    # print(model_info)
    # model = pd.read_json(sys.argv[1], encoding='ISO-8859-1')
    model = xgb.Booster(model_file=sys.argv[1])
    split_importance = model.get_score(fmap='', importance_type="weight")
    gain_importance = model.get_score(fmap='', importance_type='gain')
    feature_name = sys.argv[2].split(',')

    ft_split_importance = pd.DataFrame.from_dict({"Split Importance": split_importance.values()},
                                                 orient='Index', columns=feature_name)
    # ft_split_importance = pd.DataFrame({"Feature": feature_name, "Split Importance": split_importance})
    print(ft_split_importance)
    print()
    ft_gain_importance = pd.DataFrame.from_dict({"Gain Importance": gain_importance.values()},
                                                 orient='Index', columns=feature_name)
    # ft_gain_importance = pd.DataFrame({"Feature": feature_name, "Gain Importance": gain_importance})
    print(ft_gain_importance)

    ft_table = pd.concat([ft_split_importance, ft_gain_importance], axis=0).T
    print(ft_table)

    ft_table = ft_table.sort_values(["Split Importance", "Gain Importance"], ascending=False)

    ft_table.to_csv(sys.argv[3], sep='\t')
