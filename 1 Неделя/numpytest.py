import numpy as np
from scipy import misc

DefMatr = np.random.normal(loc = 1, scale = 10, size = (1000, 50))
Means = np.mean(DefMatr, axis = 0)
Stds = np.std(DefMatr, axis = 0)
SecMatr = (DefMatr - Means) / Stds
Means2 = np.mean(SecMatr, axis = 0)
#print(DefMatr)
#print(Means)
#print(Stds)
#print(Means2)
#print(SecMatr)

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])
Summs = np.sum(Z, axis = 1)
Logs = (Summs > 10)
TrueIndexes = np.array(np.nonzero(Logs))[0]
#print(Z)
#print(Summs)
#print(Logs)
#print(TrueIndexes)

A = np.eye(3)
B = np.eye(3)
AB = np.vstack((A, B))
print(AB)
