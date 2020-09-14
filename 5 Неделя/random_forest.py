import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

data_link = "abalone.csv"
data = pd.read_csv(data_link, delimiter=',')
# Символьное sex в число
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
# Отцепляем целевую
target = data['Rings'].to_numpy()
data = data.iloc[:,:-1].to_numpy()
# Смотрим количество деревьев от1 до 50
quality_data_list = []
for i in range(1, 51):
    regressor = RandomForestRegressor(random_state=1, n_estimators=i)
    validator = KFold(random_state=1, shuffle=True, n_splits=5)
    quality_cum = []
    for train_indices, test_indices in validator.split(data):
        # Делим выборку на обучающую и тестовую
        data_train = data[train_indices]
        data_test = data[test_indices]
        target_train = target[train_indices]
        target_test = target[test_indices]
        # Обучаемся
        regressor.fit(data_train, target_train)
        # Предсказываем
        target_pred = regressor.predict(data_test)
        # Смотрим R2-метрику
        quality_curr = round(r2_score(target_test, target_pred), 2)
        quality_cum.append(quality_curr)
    # Усредняем качество по пяти тестам
    quality_data_list.append([i, np.average(quality_cum)])
# Выводим данные
quality_data = pd.DataFrame(quality_data_list, columns=['num of trees', 'R2 score'])
print(quality_data)
