import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction import DictVectorizer

data_train_link = 'salary-train.csv'
data_test_link = 'salary-test-mini.csv'
data_train = pd.read_csv(data_train_link, delimiter=',')
data_test = pd.read_csv(data_test_link, delimiter=',')

# Нижний регистр и всё, кроме букв и цифр - в пробелы
descr_train_lowcase = data_train['FullDescription'].str.lower()
descr_test_lowcase = data_test['FullDescription'].str.lower()
data_train['FullDescription'] = descr_train_lowcase.replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['FullDescription'] = descr_test_lowcase.replace('[^a-zA-Z0-9]', ' ', regex = True)

# tfid
tfid_vectorizer = text.TfidfVectorizer(min_df=5)
X_train = tfid_vectorizer.fit_transform(data_train['FullDescription'])
X_test = tfid_vectorizer.transform(data_test['FullDescription'])

# В LocationNormalized и ContractTime заменяем пропуски на nan
# и в кучу булевых переменных
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

dict_vectorizer = DictVectorizer()

#print(data_test)
