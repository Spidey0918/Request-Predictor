# Request-Predictor

## Dataset
Since Github has a file size limit at 100MB, I put our dataset in the following link...

Link: [Request_data](https://drive.google.com/open?id=17It9Ygf50fZeqDpJE9itaNx1fTZSzeGw&authuser=haochuan.li.cn%40gmail.com&usp=drive_fs)

- Training Data: 2M (Split into train and val data at a ratio of 10:1) 
- Test Data: 1M

### Get The Data

`cd Request-Predictor`

`pip install gdown`

`gdown --no-check-certificate --folder https://drive.google.com/drive/folders/17It9Ygf50fZeqDpJE9itaNx1fTZSzeGw`

### Train LGBM

`python .\lightgbm_training.py ..\request_data\Train\TSV.tsv 3-16 9 2 ./model 50 1 20 regression 0.17 10 4 1000 0 0 0 1 0 1`

### Prediction
`python .\lightgbm_prediction.py ..\request_data\Test\TSV.tsv .\model 3-16 9 0-2 ./output.csv`

### Cal_PR_AUC
`python .\cal_pr_auc.py ..\lgbm_trainer\output.csv 3 2 ./metric.tsv ./roc_curve.pdf ./roc.tsv ./best_roc.tsv`

### Explainer
`python .\feature_importance.py ./model Request_Hour,Request_IsWeekend,Request_UserLatitude,Request_UserLongitude,Request_Count,Request_Offset,RefreshType,ImpCount1d,ClickCount1d,ImpCount3d,ClickCount3d,ImpCount7d,ClickCount7d,ActiveDays feature_importance.tsv`

`python .\shap_importance.py ..\request_data\Test\TSV.tsv .\model 3-16 9 TraceId,Request_UserId,IsShown,Request_Hour,Request_IsWeekend,Request_UserLatitude,Request_UserLongitude,Request_Count,Request_Offset,RefreshType,ImpCount1d,ClickCount1d,ImpCount3d,ClickCount3d,ImpCount7d,ClickCount7d,ActiveDays shap_importance.tsv`


