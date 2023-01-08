cd lgbm_trainer

python .\lightgbm_training.py ..\request_data\Train\TSV.tsv 3-16 9 2 ./model 50 1 20 regression 0.17 10 4 1000 0 0 0 1 0 1

python .\lightgbm_prediction.py ..\request_data\Test\TSV.tsv .\model 3-16 9 0-2 ./results/output.csv

python .\cal_pr_auc.py ..\lgbm_trainer\results\output.csv 3 2 ./results/metric.tsv ./results/roc_curve.pdf ./results/roc.tsv ./results/best_roc.tsv

python .\feature_importance.py ./model Request_Hour,Request_IsWeekend,Request_UserLatitude,Request_UserLongitude,Request_Count,Request_Offset,RefreshType,ImpCount1d,ClickCount1d,ImpCount3d,ClickCount3d,ImpCount7d,ClickCount7d,ActiveDays ./results/feature_importance.tsv

python .\shap_importance.py ..\request_data\Test\TSV.tsv .\model 3-16 9 TraceId,Request_UserId,IsShown,Request_Hour,Request_IsWeekend,Request_UserLatitude,Request_UserLongitude,Request_Count,Request_Offset,RefreshType,ImpCount1d,ClickCount1d,ImpCount3d,ClickCount3d,ImpCount7d,ClickCount7d,ActiveDays ./results/shap_importance.tsv
