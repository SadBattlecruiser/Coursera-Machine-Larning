import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn import datasets

dataAttr, dataAns = datasets.load_boston(return_X_y=True)
dataAttr = scale(dataAttr).astype('double')
#print(dataAttr.shape)
kFold = KFold(n_splits=5, shuffle=True, random_state=42)
pValues = []
qualities= []
for pVal in np.linspace(1.0, 10.0, num=200):
    knnClassifier = KNeighborsRegressor(n_neighbors=5, p=pVal, weights='distance')
    currQuality = cross_val_score(knnClassifier, dataAttr, dataAns, cv=kFold, scoring='neg_mean_squared_error')
    pValues.append(pVal)
    qualities.append(round(np.mean(currQuality), 5))
    #print('current pVal is:', pVal)
    #print('current quality:', round(np.mean(currQuality), 5), '\n')
results = pd.DataFrame({'pValues' : pValues, 'quality' : qualities})
print(results)
minIndex = results['pValues'].idxmin()
bestResult = results.loc[minIndex]
print('\nindex of best result:', minIndex,'\nbest result is:\n', bestResult)
print('----------------------------------------------------\n\n')
