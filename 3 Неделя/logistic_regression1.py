import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def dim2_grad_desc(X, y, w0_vect=[0, 0], k=0.1, eps=1e-5):
    max_steps = 10000
    w1 = w0_vect[0]
    w2 = w0_vect[1]
    L = len(y)
    #print('Gradient! Length:', L)
    for i in range(max_steps):
        w1o, w2o = w1, w2
        w1 = w1o + k/L * np.sum(y * X[:, 0] * (1 - 1/(1 + np.exp(-y*(w1o*X[:, 0] + w2o*X[:, 1])))))
        w2 = w2o + k/L * np.sum(y * X[:, 1] * (1 - 1/(1 + np.exp(-y*(w1o*X[:, 0] + w2o*X[:, 1])))))
        dist = np.linalg.norm(np.array([w1, w2] - np.array([w1o, w2o])))
        if (dist < eps):
            print('Eps! Counter:', i)
            return np.array([w1, w2])   # Да, можно было просто сделать вектора и не дергать каждый раз
    return np.array([w1, w2])

def dim2_grad_desc_norm(X, y, w0_vect=[0, 0], k=0.1, eps=1e-5, C=1):
    max_steps = 10000
    w1 = w0_vect[0]
    w2 = w0_vect[1]
    L = len(y)
    #print('Gradient! Length:', L)
    for i in range(max_steps):
        w1o, w2o = w1, w2
        w1 = w1o + k/L * np.sum(y * X[:, 0] * (1 - 1/(1 + np.exp(-y*(w1o*X[:, 0] + w2o*X[:, 1]))))) - k*C*w1o
        w2 = w2o + k/L * np.sum(y * X[:, 1] * (1 - 1/(1 + np.exp(-y*(w1o*X[:, 0] + w2o*X[:, 1]))))) - k*C*w2o
        dist = np.linalg.norm(np.array([w1, w2] - np.array([w1o, w2o])))
        if (dist < eps):
            #print('Eps! Counter:', i)
            return np.array([w1, w2])
    return np.array([w1, w2])

data_link = 'data-logistic.csv'
data_full = np.genfromtxt(data_link, delimiter=',')
data = data_full[:, 1:]
target = data_full[:, 0]
#print(data_full)
#print(data)
#print(target)
w_vec = dim2_grad_desc(data, target, k=1)
print('Non-normalized:', w_vec)
w_vec_norm = dim2_grad_desc_norm(data, target, C=10)
print('Normalized:', w_vec_norm)

scores = 1/(1 + np.exp(-w_vec[0]*data[:, 0] - w_vec[1]*data[:, 1]))
#print(scores)
scores_norm = 1/(1 + np.exp(-w_vec_norm[0]*data[:, 0] - w_vec_norm[1]*data[:, 1]))
#print(scores_norm)
quality = roc_auc_score(target, scores)
print('Quality:', round(quality, 3))
quality_norm = roc_auc_score(target, scores_norm)
print('Quality norm:', round(quality_norm, 3))
