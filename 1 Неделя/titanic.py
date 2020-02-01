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
womenFirstNamesBank = []
#   Где есть "Mrs."
isMrs = womenNames.str.find('Mrs.') > 0
mrsNames = womenNames[isMrs]
mrsBeginFirstNames = mrsNames.str.find('(') + 1
for i in range(mrsNames.shape[0] - 1):
    currFirstName = mrsNames.iloc[i][(mrsBeginFirstNames.iloc[i]) : -1]
    currFirstNameArr = currFirstName.split(' ')
    #   Надо понимать, что здесь последнее слово -- девичья фамилия. Но плевать
    for fn in currFirstNameArr:
        womenFirstNamesBank.append(fn)
#   Где есть "Miss."
missNames = womenNames[~isMrs]
missBeginFirstNames = missNames.str.find('Miss.') + 6
for i in range(missNames.shape[0] - 1):
    currFirstName = missNames.iloc[i][(missBeginFirstNames.iloc[i]) :]
    currFirstNameArr = currFirstName.split(' ')
    for fn in currFirstNameArr:
        womenFirstNamesBank.append(fn)
#print(womenFirstNamesBank)
womenFirstNamesSeries = pd.Series(womenFirstNamesBank)
popularNames = womenFirstNamesSeries.value_counts()
#print("Most popular female first names:\n", popularNames)

print(womenNames[womenNames == 'Mary'])
