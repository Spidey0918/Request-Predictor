# Start: OWN CODE
import sys
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

input_file = '../data/input_file.tsv'
predict = 1
label = 0
metric_file = '../data/metric_file.tsv'
ROC_file = '../data/ROC_file.tsv'
BestROC_file = '../data/BestROC_file.tsv'

isLocal = False

if not isLocal:
    input_file = sys.argv[1]
    predict = int(sys.argv[2])
    label = int(sys.argv[3])
    metric_file = sys.argv[4]
    ROC_file = sys.argv[6]
    BestROC_file = sys.argv[7]


y = []
scores = []
count = 0
with open(input_file, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        arr = line.split('\t')
        scores.append(float(arr[predict]))
        y.append(float(arr[label]))
        count+=1



auc = metrics.roc_auc_score(y, scores)

fpr, tpr, thresholds = metrics.roc_curve(y, scores)

tfpr = [((tpr[i] - fpr[i]) if tpr[i] > 0.9 else 0) for i in range(len(fpr))]
tfpr_array = np.array(tfpr)
max_tfpr_index = np.argmax(tfpr_array)

width = 15
height = 15
f = plt.figure(figsize=(width, height))
plt.plot(fpr, tpr, label='ROC (area = {0:.3f})'.format(auc))
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
# plt.savefig(sys.argv[2])
with PdfPages(sys.argv[5]) as pdf:
    pdf.savefig(f, bbox_inches='tight', dpi = 600)

ROC_f = open(ROC_file, 'w', encoding='utf-8')
ROC_f.write('FPR\tTPR\tThreshold\n')
for i, value in enumerate(thresholds):
    ROC_f.write('%f\t%f\t%f\n' % (fpr[i], tpr[i], value))
ROC_f.close()    

BestROC_f = open(BestROC_file, 'w', encoding='utf-8')
BestROC_f.write('FPR\tTPR\tThreshold\n')
BestROC_f.write('%f\t%f\t%f\n' % (fpr[max_tfpr_index], tpr[max_tfpr_index], thresholds[max_tfpr_index]))
BestROC_f.close() 


precision, recall, _thresholds = metrics.precision_recall_curve(y, scores)
area = metrics.auc(recall, precision)

metric_f = open(metric_file, 'w', encoding='utf-8')
metric_f.write('ROC AUC\tPR AUC\tSampleNum\n')
metric_f.write('{}\t{}\t{}\n'.format(auc, area, count))
metric_f.close()

# END: OWN CODE
