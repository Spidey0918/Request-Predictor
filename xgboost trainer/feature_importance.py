# START: OWN CODE
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import sys


def feature_importance_plot(feature_name, xlabel, importance_type, ylabel, figname):
    fig, ax = plt.subplots()
    ax.bar(feature_name, importance_type.iloc[:, 0])
    ax.set_xticklabels(labels=feature_name, rotation=45, rotation_mode="anchor", ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(figname)


if __name__ == '__main__':
    model = pickle.load(open('xgb_model.sav', 'rb'))
    gain_importance = model.feature_importances_
    # print(gain_importance)
    feature_name = sys.argv[2].split(',')
    ft_gain_importance = pd.DataFrame.from_dict({"Gain Importance": gain_importance},
                                                 orient='Index', columns=feature_name).T
    print(ft_gain_importance)
    ft_gain_importance = ft_gain_importance.sort_values("Gain Importance", ascending=False)
    feature_importance_plot(feature_name, 'feature name', ft_gain_importance,
                            'Gain importance score', 'gain_importance.png')

    ft_gain_importance.to_csv(sys.argv[3], sep='\t')
# END: OWN CODE
