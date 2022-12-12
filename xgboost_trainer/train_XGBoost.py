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


def load_data(train_data_file, val_data_file, test_data_file=None, sep='\t'):
    train_data = pd.read_csv(train_data_file, sep=sep, header=None)
    train_data.fillna(value=0, inplace = True) 
    train_data = train_data.values 

    val_data = pd.read_csv(val_data_file, sep=sep, header=None)
    val_data.fillna(value=0, inplace = True) 
    val_data = val_data.values 
    
    if test_data_file:
        test_data = pd.read_csv(test_data_file, sep=sep, header=None) 
        test_data.fillna(value=0, inplace = True) 
        test_data = test_data.values 
        
    else:
        test_data = val_data.copy()

    data_dict = {
                    'train':train_data, 
                    'val': val_data,
                    'test': test_data
                }

    return data_dict


def prepare_data(data, feature_idx_str="2:", label_idx_str="1"):
    X = eval('data[:,'+feature_idx_str+']').astype(float)
    y = eval('data[:,'+label_idx_str+']').astype(int)
    AccountIds = data[:,0]
    return X,y, AccountIds

#Currently Not Used
def train_xgb(data, params, param_comb=1, folds=5):
    X_train, y_train, AccountIds_train = prepare_data(data['train'])
    X_val, y_val, AccountIds_val = prepare_data(data['val'])

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=seed, n_jobs=1,tree_method='approx')

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = seed)
    random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=8, cv=skf.split(np.array(X_train), y_train), verbose=3, random_state=seed, refit='aucpr')
    random_search.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="aucpr", eval_set = [[X_val, y_val]])


    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)

    return random_search

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
    n_train = len(data['train'])
    n_val = len(data['val'])
    n_test = len(data['test'])

    X_train, y_train, AccountIds_train = prepare_data(data['train'])
    X_val, y_val, AccountIds_val = prepare_data(data['val'])
    X_test, y_test, AccountIds_test = prepare_data(data['test'])

    num_features = X_train.shape[1]

    train_pos_rate = y_train.mean()
    val_pos_rate = y_val.mean()
    test_pos_rate = y_test.mean()
    
    stats = {
        'n_train':n_train,
        'n_val':n_val,
        'n_test':n_test,
        'train_pos_rate':train_pos_rate,
        'val_pos_rate':val_pos_rate,
        'test_pos_rate':test_pos_rate,
        'num_features':num_features
        }
    return stats


class Tune():
    def __init__(self, data, seed):
        self.seed = seed
        self.X_train, self.y_train, self.AccountIds_train = prepare_data(data['train'])
        self.X_val, self.y_val, self.AccountIds_val = prepare_data(data['val'])
        self.X_test, self.y_test, self.AccountIds_test = prepare_data(data['test'])


    def __call__(self, trial):
        params = {}
        params['n_estimators'] = trial.suggest_categorical('n_estimators', [50, 100, 150, 200, 250, 300, 350])
        params['gamma'] = trial.suggest_float('gamma', low=0, high=2, step=1.0/50)
        params['lambda'] = trial.suggest_float('lambda', low=0, high=2, step=1.0/50)
        params['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-4, 3e-1)
        
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=self.seed, n_jobs=-1,tree_method='approx', **params)
        xgb_model.fit(self.X_train, self.y_train, early_stopping_rounds=50, eval_metric="aucpr", eval_set = [[self.X_val, self.y_val]])
               
        train_metrics = evaluate(self.X_train, self.y_train, xgb_model, self.AccountIds_train)
        val_metrics = evaluate(self.X_val, self.y_val, xgb_model, self.AccountIds_val)
        test_metrics = evaluate(self.X_test, self.y_test, xgb_model, self.AccountIds_test)
        
        print('--------------------------------------------')
        print('Trial ',trial.number)
        print('Train PR AUC - ', train_metrics['PR-AUC'])
        print('Val PR AUC - ', val_metrics['PR-AUC'])
        print('Test PR AUC - ', test_metrics['PR-AUC'])
        print('--------------------------------------------\n')
        
        trial.set_user_attr('model', xgb_model)

        for metric,value in train_metrics.items():
            trial.set_user_attr('Train ' + metric, value)

        for metric,value in val_metrics.items():
            trial.set_user_attr('Val ' + metric, value)

        for metric,value in test_metrics.items():
            trial.set_user_attr('Test ' + metric, value)
            
        return val_metrics['PR-AUC']


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--test_data', default=None, type=str)
    parser.add_argument('--n_configurations', type=int, default=250)
    parser.add_argument('--output_dir', type=str) 
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
    model_study = optuna.create_study(direction='maximize',sampler=sampler)
    model_objective = Tune(data, args.seed)
    model_study.optimize(model_objective, n_trials=args.n_configurations)
    best_trial = get_best_trial(model_study) 

    print('-------------------------------------------')
    print("Best Hyperparameters")
    print('-------------------------------------------')
    for key, value in best_trial.params.items():
        print('%s : %s'%(key,str(value)))
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

    model =  best_trial.user_attrs['model'] 
    filename = os.path.join(args.output_dir,'KeywordBaseline_XGB')
    pickle.dump(model, open(filename+'.sav', 'wb'))
   
    artefacts = {'stats:':stats,
                'model_artefacts':best_trial.user_attrs,
                'hyperparameters': best_trial.params}

    pickle.dump(artefacts, open(os.path.join(args.output_dir,'artefacts.pkl'),'wb'))

    '''
    model_onnx = convert_xgboost(model, [('input', FloatTensorType([None, stats['num_features']]))])
    with open(filename+'.onnx', "wb") as f:
        f.write(model_onnx.SerializeToString())
    '''

    