import numpy as np
import pandas as pd
from sklearn import metrics

data_cl_link = 'classification.csv'
data_cl = pd.read_csv(data_cl_link, delimiter=',')
print(data_cl)

TP_cl = (data_cl['true'] & data_cl['pred']).sum()
FP_cl = (~data_cl['true'] & data_cl['pred']).sum()
FN_cl = (data_cl['true'] & ~data_cl['pred']).sum()
TN_cl = (~data_cl['true'].astype(bool) & ~data_cl['pred']).sum()
print('TP:', TP_cl)
print('FP:', FP_cl)
print('FN:', FN_cl)
print('FT:', TN_cl)
print('SUM:', TP_cl + FP_cl + TN_cl + FN_cl)
#AP = data_cl['pred'].sum()
#AN = (~(data_cl['pred'].astype(bool))).sum()
#print(AP)
#print(AN)
print('')
accuracy = metrics.accuracy_score(data_cl['true'], data_cl['pred'])
print('ACCURACY:', round(accuracy, 3))
precision = metrics.precision_score(data_cl['true'], data_cl['pred'])
print('PRECISION:', round(precision, 3))
recall = metrics.recall_score(data_cl['true'], data_cl['pred'])
print('RECALL:', round(recall, 3))
f1 = metrics.f1_score(data_cl['true'], data_cl['pred'])
print('F1:', round(f1, 3))
