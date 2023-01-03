import os
import pickle
import sys
import optuna
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from optuna.samplers import RandomSampler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
import datetime


def get_best_trial(study):
    trials = study.trials  # .copy()
    val_metrics = np.array([t.user_attrs['Val PR-AUC'] for t in study.trials])
    best_index = np.argmax(val_metrics)
    return trials[best_index]


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
    # AccountIds = data[:, 0]
    print(X.dtypes)
    return X, y


# Currently Not Used
def train_xgb(data, params, param_comb=1, folds=5):
    X_train, y_train = prepare_data(data['train'])
    X_val, y_val = prepare_data(data['val'])

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=args.seed, n_jobs=1,
                                  tree_method='hist', enable_categorical=True)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                       n_jobs=8, cv=skf.split(np.array(X_train), y_train), verbose=3,
                                       random_state=args.seed, refit='aucpr')
    random_search.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[[X_val, y_val]])

    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)

    return random_search


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


class Tune():
    def __init__(self, data, seed):
        self.seed = seed
        self.X_train, self.y_train = prepare_data(data['train'])
        self.X_val, self.y_val = prepare_data(data['val'])
        self.X_test, self.y_test = prepare_data(data['test'])

    def __call__(self, trial):
        params = {}
        params['n_estimators'] = trial.suggest_categorical('n_estimators', [5, 10, 20, 25, 30, 40, 45, 50])
        params['gamma'] = trial.suggest_float('gamma', low=0, high=2, step=1.0 / 50)
        params['lambda'] = trial.suggest_float('lambda', low=0, high=2, step=1.0 / 50)
        params['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-4, 3e-1)

        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=self.seed, n_jobs=-1,
                                      tree_method='hist', enable_categorical=True, **params)
        xgb_model.fit(self.X_train, self.y_train, early_stopping_rounds=50, eval_metric="auc",
                      eval_set=[[self.X_val, self.y_val]], callbacks=[TensorBoardCallback()])

        train_metrics = evaluate(self.X_train, self.y_train, xgb_model)
        val_metrics = evaluate(self.X_val, self.y_val, xgb_model)
        test_metrics = evaluate(self.X_test, self.y_test, xgb_model)

        print('--------------------------------------------')
        print('Trial ', trial.number)
        print('Train ROC AUC - ', train_metrics['ROC-AUC'])
        print('Val ROC AUC - ', val_metrics['ROC-AUC'])
        print('Test ROC AUC - ', test_metrics['ROC-AUC'])
        print('--------------------------------------------\n')

        trial.set_user_attr('model', xgb_model)

        for metric, value in train_metrics.items():
            trial.set_user_attr('Train ' + metric, value)

        for metric, value in val_metrics.items():
            trial.set_user_attr('Val ' + metric, value)

        for metric, value in test_metrics.items():
            trial.set_user_attr('Test ' + metric, value)

        return val_metrics['ROC-AUC']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data', type=str, default='./TSV_train.tsv')
    parser.add_argument('--val_data', type=str, default='./TSV_test.tsv')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--n_configurations', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./model')
    args = parser.parse_args()

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

    optuna.logging.disable_default_handler()
    sampler = RandomSampler(seed=args.seed)
    model_study = optuna.create_study(direction='maximize', sampler=sampler)
    model_objective = Tune(data, args.seed)
    model_study.optimize(model_objective, n_trials=args.n_configurations)
    best_trial = get_best_trial(model_study)

    print('-----------------------------------z--------')
    print("Best Hyperparameters")
    print('-------------------------------------------')
    for key, value in best_trial.params.items():
        print('%s : %s' % (key, str(value)))
    print('-------------------------------------------')

    print('-------------------------------------------')
    print('Metrics')
    print('-------------------------------------------')
    print('Train')
    print('Accuracy', best_trial.user_attrs['Train Accuracy'])
    print('PR-AUC', best_trial.user_attrs['Train PR-AUC'])
    print('ROC-AUC', best_trial.user_attrs['Train ROC-AUC'])
    print('Classification Report')
    print(best_trial.user_attrs['Train Classification Report'])

    print()
    print('Val')
    print('Accuracy', best_trial.user_attrs['Val Accuracy'])
    print('PR-AUC', best_trial.user_attrs['Val PR-AUC'])
    print('ROC-AUC', best_trial.user_attrs['Val ROC-AUC'])
    print('Classification Report')
    print(best_trial.user_attrs['Val Classification Report'])

    print()
    print('Test')
    print('Accuracy', best_trial.user_attrs['Test Accuracy'])
    print('PR-AUC', best_trial.user_attrs['Test PR-AUC'])
    print('ROC-AUC', best_trial.user_attrs['Test ROC-AUC'])
    print('Classification Report')
    print(best_trial.user_attrs['Test Classification Report'])
    print('-------------------------------------------')

    model = best_trial.user_attrs['model']
    filename = os.path.join(args.output_dir, 'KeywordBaseline_XGB')
    pickle.dump(model, open(filename + '.sav', 'wb'))
    model.save_model('model.json')

    artefacts = {'stats:': stats,
                 'model_artefacts': best_trial.user_attrs,
                 'hyperparameters': best_trial.params}

    pickle.dump(artefacts, open(os.path.join(args.output_dir, 'artefacts.pkl'), 'wb'))

    '''
    model_onnx = convert_xgboost(model, [('input', FloatTensorType([None, stats['num_features']]))])
    with open(filename+'.onnx', "wb") as f:
        f.write(model_onnx.SerializeToString())
    '''
