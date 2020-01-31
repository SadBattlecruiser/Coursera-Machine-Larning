import pandas as pd

data_link = 'titanic_dataset.csv'
data = pd.read_csv(data_link, index_col='PassengerId')
#print(data[:10])
#print(data.head())

sexVal = data['Sex'].value_counts()
print(sexVal, '\n')


totalPassengers = data.shape[0]
survVal = data['Survived'].value_counts()
survPerc = round(survVal[1] / totalPassengers * 100, 2)
#print('Total Passengers', totalPassengers)
#print(survVal, '\n')
print('Percentage of survivors:', survPerc, '%')

classVal = data['Pclass'].value_counts()
#print(classVal)
firstClassPerc = round(classVal[1] / totalPassengers * 100, 2)
print('Percentage of first class:', firstClassPerc, '%')

ageMean = round(data['Age'].mean(axis=0), 2)
print('Mean age', ageMean)
ageMedian = round(data['Age'].median(axis=0), 2)
print('Median age', ageMedian)

correlation = round(data[['SibSp', 'Parch']].corr()['SibSp']['Parch'], 2)
print('Correlation between SibSb and Parch', correlation)

#   Часть про самое популярное женское имя
print('\n')
womenNames = data[data['Sex'] == 'female']['Name']
#print(womenNames[:10])
#print(type(womenNames))
#   Где есть "Mrs."
isMrs = womenNames.str.find('Mrs.') > 0
mrsNames = womenNames[isMrs]
mrsBeginFirstNames = mrsNames.str.find('(') + 1
#   Где есть "Miss."
missNames = womenNames[~isMrs]
missBeginFirstNames = missNames.str.find('Miss.') + 5
print('\n')
print(mrsNames)
print(mrsBeginFirstNames)
#print(missNames)
#print(missBeginFirstNames)


#print(mrsNames)
