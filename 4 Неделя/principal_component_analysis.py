import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

prices_link = "C:\\Учеба\\Машобчик\\4 Неделя\\close_prices.csv"
dow_link = "C:\\Учеба\\Машобчик\\4 Неделя\\djia_index.csv"
# Remove date-column
prices_data = pd.read_csv(prices_link, delimiter=',').iloc[:, 1:]
dow_data = pd.read_csv(dow_link, delimiter=',').iloc[:, 1:]
# Обучаем анализатор метода главных компонент
pca = PCA(n_components=10)
pca.fit(prices_data)
# Смотрим дисперсию по каждой компоненте
print("Дисперсия покомпонентно:")
print(pca.explained_variance_ratio_)
print("Дисперсия от четырех компонент:")
print(np.sum(pca.explained_variance_ratio_[:4]))
# Преобразовываем данные
prices_data_trans = pca.transform(prices_data)
# Смотрим корелляцию между первой компонентой и индексом Доу-Джонс
correlation = np.corrcoef(prices_data_trans[:, 0], dow_data.to_numpy()[:,0])
print("Коэффициент корелляции между первой компонентой и индексом Доу-Джонс: ")
print(round(correlation[0,1], 2))
# Какая компания имеет наибольший вес в первой компоненте
#print(pca.components_)
first_component = pca.components_[0, :]
max_index = np.argmax(first_component)
print("Максимальный вес в первой компоненте у компании")
print(prices_data.columns[max_index], first_component[max_index])
