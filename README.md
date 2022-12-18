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
