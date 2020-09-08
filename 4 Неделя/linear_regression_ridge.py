import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_extraction import text
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

data_train_link = 'C:\\Учеба\\Машобчик\\4 Неделя\\salary-train.csv'
data_test_link = 'C:\\Учеба\\Машобчик\\4 Неделя\\salary-test-mini.csv'
data_train = pd.read_csv(data_train_link, delimiter=',')
data_test = pd.read_csv(data_test_link, delimiter=',')

# Нижний регистр и всё, кроме букв и цифр - в пробелы
descr_train_lowcase = data_train['FullDescription'].str.lower()
descr_test_lowcase = data_test['FullDescription'].str.lower()
data_train['FullDescription'] = descr_train_lowcase.replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['FullDescription'] = descr_test_lowcase.replace('[^a-zA-Z0-9]', ' ', regex = True)

# tfid
tfid_vectorizer = text.TfidfVectorizer(min_df=5)
data_train_text = tfid_vectorizer.fit_transform(data_train['FullDescription'])
data_test_text = tfid_vectorizer.transform(data_test['FullDescription'])

# В LocationNormalized и ContractTime заменяем пропуски на nan
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

# Превращаем категориальные признаки string в наборы bool
dict_vectorizer = DictVectorizer()
data_train_categ = dict_vectorizer.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
data_test_categ = dict_vectorizer.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объединяем данные текста и категориальных признаков
data_train_sum = sp.sparse.hstack([data_train_text, data_train_categ])
data_test_sum = sp.sparse.hstack([data_test_text, data_test_categ])
print(data_train_sum.shape)
print(data_test_sum.shape)

# Обучаем регрессию
regr = Ridge(alpha=1, random_state=241)
regr.fit(data_train_sum, data_train['SalaryNormalized'])

# Прогнозы для теста
test_pred = regr.predict(data_test_sum)
print(np.round(test_pred, 2))
