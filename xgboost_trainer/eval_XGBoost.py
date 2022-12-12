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
    return X,y, AccountIds

def evaluate(X_eval, y_eval, model, AccountIds=None):
    y_pred = model.predict(np.array(X_eval))
    y_pred_proba = model.predict_proba(np.array(X_eval))[:, 1]

    cls_report = classification_report(y_eval, y_pred, digits=5)
    accuracy = accuracy_score(y_eval, y_pred)
    roc_auc = roc_auc_score(y_eval, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_eval, y_pred_proba)
    pr_auc = auc(recall, precision)

    y_pred = np.stack([AccountIds, y_pred, y_eval], axis=1)
    y_pred_proba = np.stack([AccountIds, y_pred_proba, y_eval], axis=1)

    metrics = {
        'Classification Report':cls_report,
        'Accuracy':accuracy,
        'ROC-AUC':roc_auc,
        'PR-AUC':pr_auc,
        'Predictions':y_pred,
        'Prediction Probabilities':y_pred_proba
    }
    return metrics 

def get_stats(data):
    n_eval= len(data)

    X_eval, y_eval, AccountIds_eval = prepare_data(data)

    num_features = X_eval.shape[1]

    eval_pos_rate = y_eval.mean()
    
    stats = {
        'n_eval':n_eval,
        'pos_rate':eval_pos_rate,
        'num_features':num_features
        }
    return stats


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_file', default='model.sav', type=str)
    parser.add_argument('--output_dir', type=str) 
    args = parser.parse_args()
    
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

    model = pd.read_pickle(os.path.join(args.model_dir, args.model_file))
    X_eval, y_eval, AccountIds = prepare_data(data)
    metrics = evaluate(X_eval, y_eval, model, AccountIds)
    print('-------------------------------------------')
    print('Metrics')
    print('-------------------------------------------')
    print('Accuracy', metrics['Accuracy'])
    print('PR-AUC', metrics['PR-AUC'])
    print('ROC-AUC', metrics['ROC-AUC'])
    print('Classification Report')
    print(metrics['Classification Report'])
    print()
   
    artefacts = {'stats:':stats,
                'model_artefacts':metrics
                }

    pickle.dump(artefacts, open(os.path.join(args.output_dir,'artefacts.pkl'),'wb'))

    '''
    model_onnx = convert_xgboost(model, [('input', FloatTensorType([None, stats['num_features']]))])
    with open(filename+'.onnx', "wb") as f:
        f.write(model_onnx.SerializeToString())
    '''

    