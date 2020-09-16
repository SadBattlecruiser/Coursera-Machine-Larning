import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

data_link = 'gbm-data.csv'
data = pd.read_csv(data_link, delimiter=',')
# Отщепляем целевой класс
target = data['Activity']
data = data.iloc[:, 1:]
# Делим выборку на обучающую и тестовую
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.8, random_state=241)
# Обучаемся с различными learning_rate
learning_rate_list = [1, 0.5, 0.3, 0.2, 0.1]
gb_classifier_list = []
for lr in learning_rate_list:
    print("current learning rate is:", lr)
    gb_classifier = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    gb_classifier.fit(data_train, target_train)
    gb_classifier_list.append([lr, gb_classifier])
# Смотрим ошибки на тестовой выборке
lr_loss_list = []
for lr, gb_classifier in gb_classifier_list:
    #print("current learning rate is:", lr)
    loss_iter_list = []
    # Ошибка по итерациям
    for i, target_pred in enumerate(gb_classifier.staged_decision_function(data_test)):
        # target-pred - массив с decision_function, т.е. качество
        # Пересчитываем в вероятности по сигмойде
        # Вместо этого можно было сразу staged_predict_proba
        target_pred = 1. / (1. + np.exp(-target_pred))
        # Смотрим log-loss
        loss = log_loss(target_test, target_pred)
        loss_iter_list.append([i + 1, loss])
    loss = pd.DataFrame(loss_iter_list, columns=['iter','log loss'])
    #print(loss)
    lr_loss_list.append([lr, loss])
# Строим графики ошибок на тестовой по итерациям
fig = plt.figure(figsize=(15,8))
fig.suptitle('log_loss по итерациям для разных learning_rate на тестовой выборке', fontsize=16)
for lr, loss in lr_loss_list:
    plt.plot(loss['iter'], loss['log loss'], label=str(lr))
fig.legend(loc='upper left')
plt.show()
# Ищем наилучшее количество итераций для learn_rate 0.2
for lr, loss in lr_loss_list:
    if lr == 0.2:
        index_of_min = loss['log loss'].idxmin()
        best = loss.iloc[index_of_min]
        print('Best result for learn_rate 0.2:')
        print(best)
# Обучаем RF с лучшими для GB параметрами
rf_classifier = GradientBoostingClassifier(random_state=241, n_estimators=np.int(best['iter']))
rf_classifier.fit(data_train, target_train)
# Ошибка RF
rf_target_pred = rf_classifier.predict_proba(data_test)
rf_loss = log_loss(target_test, rf_target_pred)
print('RF log loss is:')
print(rf_loss)
