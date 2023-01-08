# START: OWN CODE
import pandas as pd
from class_balancing import balance_data
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, auc, log_loss, precision_recall_curve, accuracy_score, classification_report
from parsing_utility import index_range_parser
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgbm
import sys
import json

# Tune Parameters:
# Tree Structure: num_leaves, min_data_in_leaf, max_depth
# Accuracy: n_estimators, learning_rate
# Control overfitting: reg_lambda, reg_alpha, min_split_gain

input_data = pd.read_csv(sys.argv[1], header=None, sep='\t')
feature_indices = index_range_parser(sys.argv[2])
label_ind = int(sys.argv[4])
n_estimators = index_range_parser(sys.argv[7])
verbose = int(sys.argv[8])
early_stopping_rounds = index_range_parser(sys.argv[9])
model_objective = sys.argv[10]
learning_rate = [eval(index.strip()) for index in sys.argv[12].split(',')]
num_leaves = [eval(index.strip()) for index in sys.argv[13].split(',')]
max_depth = [eval(index.strip()) for index in sys.argv[14].split(',')]
min_data_in_leaf = [eval(index.strip()) for index in sys.argv[15].split(',')]
reg_alpha = [eval(index.strip()) for index in sys.argv[16].split(',')]
reg_lambda = [eval(index.strip()) for index in sys.argv[17].split(',')]
min_gain_to_split = [eval(index.strip()) for index in sys.argv[18].split(',')]
subsample = [eval(index.strip()) for index in sys.argv[19].split(',')]
subsample_freq = index_range_parser(sys.argv[20])
colsample_bytree = [eval(index.strip()) for index in sys.argv[21].split(',')]


# category_indices == '-' means no categorical features
if sys.argv[3] != '-':
    category_indices = index_range_parser(sys.argv[3])
    input_data = balance_data(input_data, label_ind, category_indices)
outputs = sys.argv[5]
output_param = sys.argv[6]

features = input_data.iloc[:, feature_indices]
labels = input_data.iloc[:, label_ind]


def get_best_trial(study):
    trials = study.trials.copy()
    # print([t for t in study.trials])
    val_metrics = np.array([t.user_attrs['Val ROC-AUC']
                           if 'Val ROC-AUC' in t.user_attrs else 0 for t in study.trials])
    best_index = np.argmax(val_metrics)
    return trials[best_index]


def evaluate(X_eval, y_eval, model):
    y_pred = model.predict(X_eval)
    # y_pred_proba = model.predict_proba(np.array(X_eval))[:, 1]

    roc_auc = roc_auc_score(y_eval, y_pred)
    precision, recall, _ = precision_recall_curve(y_eval, y_pred)
    pr_auc = auc(recall, precision)

    y_pred = np.stack([y_pred, y_eval], axis=1)
    # y_pred_proba = np.stack([y_pred_proba, y_eval], axis=1)

    metrics = {
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Predictions': y_pred

    }
    return metrics


def lightgbm_training(features, labels, outputs, n_estimators, verbose, early_stopping_rounds, objective):
    features, features_test, labels, labels_test = train_test_split(
        features, labels, test_size=0.1)

    model = lgbm.LGBMClassifier(n_estimators=n_estimators, objective=objective)
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


def objective(trial):
    param_grid = {
        "objective": model_objective,
        "metric": "auc, binary_logloss",
        "verbose": -1,
        "feature_pre_filter": False,
        "boosting_type": 'gbdt',
        "n_estimators": trial.suggest_categorical("n_estimators", n_estimators),
        # "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 3e-1),
        "learning_rate": trial.suggest_loguniform("learning_rate", learning_rate[0], learning_rate[1]),
        # "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "num_leaves": trial.suggest_int("num_leaves", num_leaves[0], num_leaves[1], step=num_leaves[2]),
        # "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
        "max_depth": trial.suggest_int("max_depth", max_depth[0], max_depth[1], step=max_depth[2]),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", min_data_in_leaf[0], min_data_in_leaf[1], step=min_data_in_leaf[2]),
        # "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
        "reg_alpha": trial.suggest_int("reg_alpha", reg_alpha[0], reg_alpha[1], step=reg_alpha[2]),
        # "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),
        "reg_lambda": trial.suggest_int("reg_lambda", reg_lambda[0], reg_lambda[1], step=reg_lambda[2]),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15, step=1),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", min_gain_to_split[0], min_gain_to_split[1], step=min_gain_to_split[2]),
        # "subsample": trial.suggest_float("subsample", 0.2, 0.95, step=0.05),
        "subsample": trial.suggest_float("subsample", subsample[0], subsample[1], step=subsample[2]),
        "subsample_freq": trial.suggest_categorical("subsample_freq", subsample_freq),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.95, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", colsample_bytree[0], colsample_bytree[1], step=colsample_bytree[2]),
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", early_stopping_rounds),
        "random_state": 2022,
    }

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.1, shuffle=True)
    dtrain = lgbm.Dataset(features_train, label=labels_train)
    dvalid = lgbm.Dataset(features_test, label=labels_test)

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    model = lgbm.train(param_grid, dtrain, valid_sets=[
                       dvalid], callbacks=[pruning_callback])

    train_metrics = evaluate(features_train, labels_train, model)
    val_metrics = evaluate(features_test, labels_test, model)

    print('--------------------------------------------')
    print('Trial ', trial.number)
    print('Train ROC-AUC - ', train_metrics['ROC-AUC'])
    print('Val ROC-AUC - ', val_metrics['ROC-AUC'])
    print('--------------------------------------------\n')

    trial.set_user_attr('model', model)

    for metric, value in train_metrics.items():
        trial.set_user_attr('Train ' + metric, value)

    for metric, value in val_metrics.items():
        trial.set_user_attr('Val ' + metric, value)

    return val_metrics['ROC-AUC']


if __name__ == '__main__':
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize"
    )
    trials = int(sys.argv[11])

    study.optimize(objective, n_trials=trials)
    # func = lambda trial : objective(trial, input_data.iloc[:, feature_indices], input_data.iloc[:, label_ind])
    best_trial = get_best_trial(study)

    with open(output_param, 'w') as f:
        f.write(json.dumps(best_trial.params))

    print('-------------------------------------------')
    print("Best Hyperparameters")
    print('-------------------------------------------')
    for key, value in best_trial.params.items():
        print('%s : %s' % (key, str(value)))
    print('-------------------------------------------')

    print('-------------------------------------------')
    print('Metrics')
    print('-------------------------------------------')
    print('Train')
    print('PR-AUC', best_trial.user_attrs['Train PR-AUC'])
    print('ROC-AUC', best_trial.user_attrs['Train ROC-AUC'])

    print()
    print('Val')
    print('PR-AUC', best_trial.user_attrs['Val PR-AUC'])
    print('ROC-AUC', best_trial.user_attrs['Val ROC-AUC'])

    model = best_trial.user_attrs['model']
    model.save_model(outputs)

# END: OWN CODE
