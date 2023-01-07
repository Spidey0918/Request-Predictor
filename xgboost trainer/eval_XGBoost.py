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
from sklearn.metrics import precision_recall_curve


def get_best_trial(study):
    trials = study.trials.copy()
    val_metrics = np.array([t.user_attrs['Val PR-AUC'] for t in study.trials])
    best_index = np.argmax(val_metrics)
    return trials[best_index]


def load_data(eval_data_file, sep='\t'):
    eval_data = pd.read_csv(eval_data_file, sep=sep, header=None)
    eval_data.fillna(value=0, inplace = True) 
    eval_data = eval_data.values 
    return eval_data


def prepare_data(data, feature_idx_str="2:", label_idx_str="1"):
    X = eval('data[:,'+feature_idx_str+']').astype(float)
    y = eval('data[:,'+label_idx_str+']').astype(int)
    AccountIds = data[:,0]
    return X, y, AccountIds


def evaluate(X_eval, y_eval, model):
    X_eval.iloc[:, 6] = X_eval.iloc[:, 6].astype("category")
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)[:, 1]

    cls_report = classification_report(y_eval, y_pred, digits=5)
    conf_matrix = confusion_matrix(y_eval, y_pred)
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
        'Prediction Probabilities': y_pred_proba,
        'confusion matrix': conf_matrix
    }

    # print("classification_report:\n", cls_report)
    # print("confusion matrix:\n", conf_matrix)z
    return metrics


def get_stats(data):
    n_eval = len(data)
    X_eval, y_eval, AccountIds_eval = prepare_data(data)
    num_features = X_eval.shape[1]
    eval_pos_rate = y_eval.mean()
    
    stats = {
        'n_eval':n_eval,
        'pos_rate':eval_pos_rate,
        'num_features':num_features
        }

    return stats


# start: NOT own code
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
# end: NOT own code


if __name__ =='__main__':
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
    data = load_data(args.eval_data)
    stats = get_stats(data)

    print('-------------------------------------------')
    print('Data Statistics')
    print('-------------------------------------------')
    print('No. of Eval Accounts - ', stats['n_eval'])
    print()
    print('Number of Features -', stats['num_features'])
    print()
    print('Positive Rate - ', stats['pos_rate'])
    print('-------------------------------------------')

    # -------------------------------------------
    # Best Hyperparameters (from the former model training step)
    # -------------------------------------------
    # n_estimators: 50
    # gamma: 1.68
    # lambda: 0.24
    # learning_rate: 0.15917647476712105
    # -------------------------------------------

    # optional: train xgb model using Best Hyper-parameters totally with Tensorboard
    # params = {'n_estimators': 50, 'gamma': 1.68, 'lambda': 0.24, 'learning_rate': 0.15917647476712105}
    # model = train_xgb(data, params)
    # filename = os.path.join(args.output_dir, 'model_xgb')
    # pickle.dump(model, open(filename + '.sav', 'wb'))
    # model.save_model('model_xgb.json')

    # Recommended: load model directly
    model = pickle.load(open('model_xgb.sav', 'rb'))
    X_eval, y_eval, AccountIds = prepare_data(data)
    metrics = evaluate(X_eval, y_eval, model)
    print('-------------------------------------------')
    print('Metrics')
    print('-------------------------------------------')
    print('Accuracy', metrics['Accuracy'])
    print('PR-AUC', metrics['PR-AUC'])
    print('ROC-AUC', metrics['ROC-AUC'])
    print('Classification Report')
    print(metrics['Classification Report'])
    print('Confusion Matrix')
    print(metrics['Confusion Matrix'])
    print()

    print('Done!')
