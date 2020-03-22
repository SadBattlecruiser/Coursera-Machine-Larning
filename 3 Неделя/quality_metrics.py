import numpy as np
import pandas as pd
from sklearn import metrics

data_cl_link = 'classification.csv'
data_cl = pd.read_csv(data_cl_link, delimiter=',')
#print(data_cl)

TP_cl = (data_cl['true'] & data_cl['pred']).sum()
FP_cl = (~data_cl['true'] & data_cl['pred']).sum()
FN_cl = (data_cl['true'] & ~data_cl['pred']).sum()
TN_cl = (~data_cl['true'].astype(bool) & ~data_cl['pred']).sum()
#print('TP:', TP_cl)
#print('FP:', FP_cl)
#print('FN:', FN_cl)
#print('FT:', TN_cl)
#print('SUM:', TP_cl + FP_cl + TN_cl + FN_cl, '\n')
accuracy = metrics.accuracy_score(data_cl['true'], data_cl['pred'])
#print('ACCURACY:', round(accuracy, 3))
precision = metrics.precision_score(data_cl['true'], data_cl['pred'])
#print('PRECISION:', round(precision, 3))
recall = metrics.recall_score(data_cl['true'], data_cl['pred'])
#print('RECALL:', round(recall, 3))
f1 = metrics.f1_score(data_cl['true'], data_cl['pred'])
#print('F1:', round(f1, 3))


data_sc_link = 'scores.csv'
data_sc = pd.read_csv(data_sc_link, delimiter=',')
print(data_sc)
logreg_roc_auc = metrics.roc_auc_score(data_sc['true'], data_sc['score_logreg'])
print('LOGREG ROC-AUC:', round(logreg_roc_auc, 3))
svm_roc_auc = metrics.roc_auc_score(data_sc['true'], data_sc['score_svm'])
print('SVM ROC-AUC:', round(svm_roc_auc, 3))
knn_roc_auc = metrics.roc_auc_score(data_sc['true'], data_sc['score_knn'])
print('KNN ROC-AUC:', round(knn_roc_auc, 3))
tree_roc_auc = metrics.roc_auc_score(data_sc['true'], data_sc['score_tree'])
print('TREE ROC-AUC:', round(tree_roc_auc, 3), '\n')

# Точки ROC-кривой для логистической регрессии
logreg_prec, logreg_rec, logreg_thres = metrics.precision_recall_curve(
                            data_sc['true'], data_sc['score_logreg'])
# Те значения точности, где полнота больше 70%
logreg_prec_good = logreg_prec[logreg_rec >= 0.7]
logreg_prec_max = np.max(logreg_prec_good)
print('LOGREG MAX PREC:', round(logreg_prec_max, 3))

# Аналогично для остальных
svm_prec, svm_rec, svm_thres = metrics.precision_recall_curve(
                            data_sc['true'], data_sc['score_svm'])
svm_prec_good = svm_prec[svm_rec >= 0.7]
svm_prec_max = np.max(svm_prec_good)
print('SVM MAX PREC:', round(svm_prec_max, 3))

knn_prec, knn_rec, knn_thres = metrics.precision_recall_curve(
                            data_sc['true'], data_sc['score_knn'])
knn_prec_good = knn_prec[knn_rec >= 0.7]
knn_prec_max = np.max(knn_prec_good)
print('KNN MAX PREC:', round(knn_prec_max, 3))

tree_prec, tree_rec, tree_thres = metrics.precision_recall_curve(
                            data_sc['true'], data_sc['score_tree'])
tree_prec_good = tree_prec[tree_rec >= 0.7]
tree_prec_max = np.max(tree_prec_good)
print('TREE MAX PREC:', round(tree_prec_max, 3))
