import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#   Задача на тестирование перцептрона
dataTrainLink = 'perceptron-train.csv'
dataTestLink = 'perceptron-test.csv'
colNames = ['target', 'attr1', 'attr2']
dataTrain = pd.read_csv(dataTrainLink, header=None, index_col=False, names=colNames)
dataTest = pd.read_csv(dataTestLink, header=None, index_col=False, names=colNames)
dataTrainAttr = dataTrain[colNames[1:]]
dataTrainAns = dataTrain['target']
dataTestAttr = dataTest[colNames[1:]]
dataTestnAns = dataTest['target']

perceptron = Perceptron()
perceptron.fit(dataTrainAttr, dataTrainAns)
predicts = perceptron.predict(dataTestAttr)
quality = accuracy_score(dataTestnAns.to_numpy(), predicts)
#print(predicts)
#print(dataTestnAns.to_numpy())
print('non-scaled quality is:', quality)

#   То же самое,но с масштабированными признаками
scaler = StandardScaler()
dataTrainAttrScaled = scaler.fit_transform(dataTrainAttr)
dataTestAttrScaled = scaler.transform(dataTestAttr)
perceptron.fit(dataTrainAttrScaled, dataTrainAns)
predictsScaled = perceptron.predict(dataTestAttrScaled)
qualityScaled = accuracy_score(dataTestnAns.to_numpy(), predictsScaled)
print('scaled quality is:', qualityScaled)
print('difference is:', round(qualityScaled - quality, 3))
#print(dataTrainAttrScaled)
