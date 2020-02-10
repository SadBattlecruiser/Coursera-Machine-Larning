import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#   Задача про предсказание сорта винограда
dataLink = 'wine.data'
attrNames = ['Sort','Alcohol', 'Malic acid', 'Ash', 'Alc. of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonfl. phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
dataFull = pd.read_csv(dataLink, header=None, names=attrNames, index_col=False)
dataAns = dataFull['Sort']
dataAttr = dataFull[attrNames[1:]]
#dataFull.to_csv('test.txt')
#print(dataFull)
