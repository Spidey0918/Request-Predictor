import pandas as pd
from class_balancing import balance_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from parsing_utility import index_range_parser
import lightgbm as lgbm
import sys


def lightgbm_training(features, labels, outputs, n_estimators, verbose, early_stopping_rounds, objective,
                      learning_rate, num_leaves, max_depth, min_data_in_leaf,
                      reg_alpha, reg_lambda, min_gain_to_split, subsample,
                      subsample_freq, colsample_bytree):
    features, features_test, labels, labels_test = train_test_split(
        features, labels, test_size=0.1)

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "min_gain_to_split": min_gain_to_split,
        "subsample": subsample,
        "subsample_freq": subsample_freq,
        "colsample_bytree": colsample_bytree,
        # "early_stopping_rounds": early_stopping_rounds
    }

    model = lgbm.LGBMClassifier(objective=objective, **params)
    model.fit(features, labels, eval_set=[
              (features_test, labels_test)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    model.booster_.save_model(outputs)

    feature_importances = model.booster_.feature_importance(
        importance_type='gain')
    normalized_importances = feature_importances / sum(feature_importances)
    sorted_features = np.argsort(normalized_importances * (-1))
    feature_list = features.columns
    for i in range(0, min(10, len(sorted_features))):
        print('SystemLog: {} {:.2f}'.format(feature_list[sorted_features[i]],
                                            normalized_importances[sorted_features[i]]))
    print('SystemLog: validation ROCAUC ', roc_auc_score(
        labels_test, model.predict_proba(features_test)[:, 1]))


if __name__ == '__main__':
    input_data = pd.read_csv(sys.argv[1], header=None, sep='\t')
    feature_indices = index_range_parser(sys.argv[2])
    label_ind = int(sys.argv[4])
    n_estimators = int(sys.argv[6])
    verbose = int(sys.argv[7])
    early_stopping_rounds = int(sys.argv[8])
    objective = sys.argv[9]
    learning_rate = float(sys.argv[10])
    num_leaves = int(sys.argv[11])
    max_depth = int(sys.argv[12])
    min_data_in_leaf = int(sys.argv[13])
    reg_alpha = int(sys.argv[14])
    reg_lambda = int(sys.argv[15])
    min_gain_to_split = int(sys.argv[16])
    subsample = float(sys.argv[17])
    subsample_freq = int(sys.argv[18])
    colsample_bytree = float(sys.argv[19])

    # category_indices == '-' means no categorical features
    if sys.argv[3] != '-':
        category_indices = index_range_parser(sys.argv[3])
        input_data = balance_data(input_data, label_ind, category_indices)

    lightgbm_training(input_data.iloc[:, feature_indices], input_data.iloc[:, label_ind],
                      sys.argv[5], n_estimators, verbose, early_stopping_rounds, objective,
                      learning_rate, num_leaves, max_depth, min_data_in_leaf,
                      reg_alpha, reg_lambda, min_gain_to_split, subsample,
                      subsample_freq, colsample_bytree)
