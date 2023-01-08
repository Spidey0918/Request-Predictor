import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import sys
from parsing_utility import index_range_parser



if __name__ == '__main__':
    model = lgb.Booster(model_file=sys.argv[1])
    split_importance = model.feature_importance(importance_type="split")
    gain_importance = model.feature_importance(importance_type='gain')
    feature_name = sys.argv[2].split(',')
    
    print(feature_name)
    ft_split_importance = pd.DataFrame({"Feature":feature_name, "Split Importance":split_importance})
    
    ft_gain_importance = pd.DataFrame({"Feature":feature_name, "Gain Importance":gain_importance})
  
    
    ft_table = pd.concat([ft_split_importance, ft_gain_importance], axis=1)
    ft_table = ft_table.sort_values(["Split Importance", "Gain Importance"], ascending=False)

    ft_table.to_csv(sys.argv[3], sep='\t')