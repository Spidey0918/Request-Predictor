import os
import pickle
import sys
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from tensorboardX import SummaryWriter
import datetime


def load_data(train_data_file, val_data_file, test_data_file=None, sep='\t'):
    train_data = pd.read_csv(train_data_file, sep=sep, header=None)
    # train_data.fillna(value=0, inplace=True)
    # train_data = train_data
    print(train_data.shape)

    val_data = pd.read_csv(val_data_file, sep=sep, header=None)
    # val_data.fillna(value=0, inplace=True)
    # val_data = val_data

    if test_data_file:
        test_data = pd.read_csv(test_data_file, sep=sep, header=None)
        # test_data.fillna(value=0, inplace=True)
        test_data = test_data

    else:
        test_data = val_data.copy()

    data_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    return data_dict


def prepare_data(data, feature_idx=[3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16], label_idx_str="2"):
    y = eval('data.iloc[:,' + label_idx_str + ']').astype(int)
    X = data.iloc[:, list(range(3, 17))]
    feature_idx = [x - 3 for x in feature_idx]
    X.iloc[:, feature_idx] = eval('X.iloc[:,' + str(feature_idx) + ']').astype(float)
    X.iloc[:, 6] = eval('X.iloc[:,6]').astype("category")
    return X, y


def train_xgb(data, params):
    X_train, y_train = prepare_data(data['train'])
    X_val, y_val = prepare_data(data['val'])

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=args.seed, n_jobs=1,
                                  tree_method='hist', enable_categorical=True, **params)
    xgb_model.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc",
                      eval_set=[[X_val, y_val]],
                      callbacks=[TensorBoardCallback(experiment='xgboost_exp', data_name='train')])
    print(xgb_model)

    return xgb_model


def evaluate(X_eval, y_eval, model):
    X_eval.iloc[:, 6] = X_eval.iloc[:, 6].astype("category")
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)[:, 1]

    cls_report = classification_report(y_eval, y_pred, digits=5)
    accuracy = accuracy_score(y_eval, y_pred)
    roc_auc = roc_auc_score(y_eval, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_eval, y_pred_proba)
    pr_auc = auc(recall, precision)

    y_pred = np.stack([y_pred, y_eval], axis=1)
    y_pred_proba = np.stack([y_pred_proba, y_eval], axis=1)

    metrics = {
        'Classification Report': cls_report,
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Predictions': y_pred,
        'Prediction Probabilities': y_pred_proba
    }
    return metrics


def get_stats(data):
    n_train = len(data['train'])
    n_val = len(data['val'])
    n_test = len(data['test'])

    X_train, y_train = prepare_data(data['train'])
    X_val, y_val = prepare_data(data['val'])
    X_test, y_test = prepare_data(data['test'])

    num_features = X_train.shape[1]

    train_pos_rate = y_train.mean()
    val_pos_rate = y_val.mean()
    test_pos_rate = y_test.mean()

    stats = {
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'train_pos_rate': train_pos_rate,
        'val_pos_rate': val_pos_rate,
        'test_pos_rate': test_pos_rate,
        'num_features': num_features
    }
    return stats


class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, experiment: str = None, data_name: str = None):
        self.experiment = experiment or "logs"
        self.data_name = data_name or "test"
        self.datetime_ = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"runs/{self.experiment}/{self.datetime_}"
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "train/"))
        if self.data_name:
            self.test_writer = SummaryWriter(
                log_dir=os.path.join(self.log_dir, f"{self.data_name}/")
            )

    def after_iteration(
        self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.train_writer.add_scalar(metric_name, score, epoch)
                else:
                    self.test_writer.add_scalar(metric_name, score, epoch)

        return False


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data', type=str, default='./TSV_train.tsv')
    parser.add_argument('--val_data', type=str, default='./TSV_test.tsv')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--n_configurations', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./model')
    args = parser.parse_args()

    # output dir and load data
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.train_data, args.val_data, args.test_data)
    stats = get_stats(data)

    print('-------------------------------------------')
    print('Data Statistics')
    print('-------------------------------------------')
    print('No. of Train Records - ', stats['n_train'])
    print('No. of Val Records - ', stats['n_val'])
    print('No. of Test Records - ', stats['n_test'])
    print()
    print('Number of Features -', stats['num_features'])
    print()
    print('Train Positive Rate - ', stats['train_pos_rate'])
    print('Val Positive Rate - ', stats['val_pos_rate'])
    print('Test Positive Rate - ', stats['test_pos_rate'])
    print('-------------------------------------------')

    # -----------------------------------z--------
    # Best Hyperparameters
    # -------------------------------------------
    # n_estimators : 200
    # gamma : 0.56
    # lambda : 1.9000000000000001
    # learning_rate : 0.12461063963087737
    # -------------------------------------------

    # do not forget to change the best params gained from optuna
    params = {'n_estimators': 200, 'gamma': 0.56, 'lambda': 1.9, 'learning_rate': 0.12461063963087737}
    model = train_xgb(data, params)
    model.save_model('model.json')

    X_test, y_test = prepare_data(data['test'])
    test_metric = evaluate(X_test, y_test, model)
    print('ROC_AUC:', test_metric['ROC-AUC'])

    print('Done!\n')

