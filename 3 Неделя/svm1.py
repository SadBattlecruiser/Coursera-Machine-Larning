import numpy as np
import pandas as pd
from sklearn.svm import SVC

data_link = 'svm-data.csv'
data = np.genfromtxt(data_link, delimiter=',')
x_data = data[:, 1:]
y_data = data[:, 0]
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(x_data, y_data)

print(data)
print(x_data)
print(y_data)
print(clf.support_)
