import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC

#   Задачка про распознавание категории текста
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
#print(newsgroups)
