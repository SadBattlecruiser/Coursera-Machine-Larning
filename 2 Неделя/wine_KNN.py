import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

#   Задача про предсказание сорта винограда
dataLink = 'wine.data'
attrNames = ['Sort','Alcohol', 'Malic acid', 'Ash', 'Alc. of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonfl. phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
dataFull = pd.read_csv(dataLink, header=None, names=attrNames, index_col=False)
dataAns = dataFull['Sort']
dataAttr = dataFull[attrNames[1:]]
#   Фигачим кросс-валидацию
#   Это штука, которая разбивает датасет на выборки
kFold = KFold(n_splits=5, shuffle=True, random_state=42)
#   Подбираем оптимальное количество соседей
for neighNum in range(1, 51, 1):
    valIndexes = kFold.split(dataAns)
    #   Это сам классификатор
    knnClassifier = KNeighborsClassifier(n_neighbors=neighNum)
    #print('current classifier params is:', knnClassifier.get_params())
    #   Кросс-валидация для текущего neighNum
    totalRightAns = 0
    totalAns = 0
    for train_index, test_index in valIndexes:
        #   Текущая обучающая и тестовая выборки параметров и ответов
        dataAttrTrain, dataAttrTest = dataAttr.loc[train_index], dataAttr.loc[test_index]
        dataAnsTrain, dataAnsTest = dataAns[train_index], dataAns[test_index]
        #   Обучаемся на текущей выборке
        knnClassifier.fit(dataAttrTrain, dataAnsTrain)
        #   Предсказываем тест и запоминаем количество правильных ответов
        predicts = knnClassifier.predict(dataAttrTest)
        #print(' predicts:\n', predicts, '\n', 'answers:\n', dataAnsTest.to_numpy())
        isRight = (predicts == dataAnsTest.to_numpy()).astype(int)
        #print(' isRight:\n', isRight)
        totalRightAns += np.sum(isRight)
        totalAns += isRight.shape[0]
        #print(' totalRightAns:', totalRightAns, 'totalAns:', totalAns)
        #print('\n\n')
    print('current neighNum is:', neighNum)
    print('ratio of correct answers:', round(totalRightAns / totalAns, 2), '\n')
print('----------------------------------------------------\n\n')
