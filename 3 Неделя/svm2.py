import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#   Задачка про распознавание категории текста
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
vectorizer = TfidfVectorizer()      # Штука, которая делает TF-IDF по текстам
validator = KFold(n_splits=5, shuffle=True, random_state=241)
classifier = SVC(kernel='linear', random_state=241)
param_arr =  np.power(10.0, np.arange(-5, 6))
param_grid = {'C': param_arr}

vectorized_data = vectorizer.fit_transform(newsgroups.data)
gs = GridSearchCV(classifier, param_grid, scoring='accuracy', cv=validator, n_jobs=3)
gs.fit(vectorized_data, newsgroups.target)
best_param = gs.best_params_['C']
print('best parameter C is:', best_param)
best_classifier = SVC(kernel='linear', random_state=241, C=best_param)
best_classifier.fit(vectorized_data, newsgroups.target)
coeffs = best_classifier.coef_.todense().tolist()[0]
coeffs2 = np.absolute(coeffs)
indexes_of_max_coeffs = np.argsort(coeffs2)[-10:]     # Индексы 10-и слов с самыми большими коэффициентами
print('10 max indexes:', len(indexes_of_max_coeffs))
feature_mapping = np.array(vectorizer.get_feature_names())
words = np.sort(feature_mapping[indexes_of_max_coeffs])
print('words:', words)
