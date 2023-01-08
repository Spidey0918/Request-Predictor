python ./lgbm_trainer/lightgbm_training.py ./request_data/Train/TSV.tsv 3-16 9 2 ./lgbm_trainer/model 50 1 20 regression 0.17 10 4 1000 0 0 0 1 0 1
echo ------------------
python ./lgbm_trainer/lightgbm_prediction.py ./request_data/Test/TSV.tsv ./lgbm_trainer/model 3-16 9 0-2 ./lgbm_trainer/results/output.csv
echo ------------------
python ./lgbm_trainer/cal_pr_auc.py ./lgbm_trainer/results/output.csv 3 2 ./lgbm_trainer/results/metric.tsv ./lgbm_trainer/results/roc_curve.pdf ./lgbm_trainer/results/roc.tsv ./lgbm_trainer/results/best_roc.tsv
echo ------------------
python ./lgbm_trainer/feature_importance.py ./lgbm_trainer/model Request_Hour,Request_IsWeekend,Request_UserLatitude,Request_UserLongitude,Request_Count,Request_Offset,RefreshType,ImpCount1d,ClickCount1d,ImpCount3d,ClickCount3d,ImpCount7d,ClickCount7d,ActiveDays ./lgbm_trainer/results/feature_importance.tsv
echo ------------------
python ./lgbm_trainer/shap_importance.py ./request_data/Test/TSV.tsv ./lgbm_trainer/model 3-16 9 TraceId,Request_UserId,IsShown,Request_Hour,Request_IsWeekend,Request_UserLatitude,Request_UserLongitude,Request_Count,Request_Offset,RefreshType,ImpCount1d,ClickCount1d,ImpCount3d,ClickCount3d,ImpCount7d,ClickCount7d,ActiveDays ./lgbm_trainer/results/shap_importance.tsv
echo ------------------
