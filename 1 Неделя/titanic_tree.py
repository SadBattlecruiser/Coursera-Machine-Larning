import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data_link = 'titanic_dataset.csv'
#   Решаем задачку про важность признаков для решающего дерева
dataFull = pd.read_csv(data_link, index_col='PassengerId')[['Pclass', 'Fare', 'Age', 'Sex','Survived']]
dataFullClear = dataFull[~dataFull['Age'].isna()]
data = dataFullClear[['Pclass', 'Fare', 'Age', 'Sex']]
survAns = dataFullClear['Survived']
#   Делаем пол 1 или 0
isMale = (data['Sex'] == 'male').astype(int)
data['Sex'] = isMale
print(data)
print(survAns)
#   Конвертируем в массивы numpy (не очень-то и нужно)
dataNP = data.to_numpy()
survAnsNP = survAns.to_numpy()
#print(dataNP)
#   Обучаемся
treeCls = DecisionTreeClassifier()
treeCls.random_state = 241
treeCls.fit(data, survAns)
#   Смотрим признаки
importances = treeCls.feature_importances_
print(importances)
