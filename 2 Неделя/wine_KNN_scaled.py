import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
#   Теперь то же самое, только пронормировав признаки и с помощью встроенной функции
#   Костыли с превращением numpy array в pandas dataframe чтобы потренироваться
#print(dataFull)
dataLink = 'wine.data'
attrNames = ['Sort','Alcohol', 'Malic acid', 'Ash', 'Alc. of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonfl. phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
dataFull = pd.read_csv(dataLink, header=None, names=attrNames, index_col=False)
dataAns = dataFull['Sort']
dataAttrTemp = dataFull[attrNames[1:]]
dataAttr = pd.DataFrame(scale(dataAttrTemp)).astype('double')
print(dataFull)
#print(dataFull)
print(dataAns)
kFold = KFold(n_splits=5, shuffle=True, random_state=42)
for neighNum in range(1, 51, 1):
    knnClassifier = KNeighborsClassifier(n_neighbors=neighNum)
    currQuality = cross_val_score(knnClassifier, dataAttr, dataAns, cv=kFold, scoring='accuracy')
    print('current neighNum is:', neighNum)
    print('ratio of correct answers:', round(np.mean(currQuality), 5), '\n')
print('----------------------------------------------------\n\n')
