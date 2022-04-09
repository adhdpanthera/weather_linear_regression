import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Считываем csv-таблицу в переменную data, удаляем первые 6 строк комментарией
data = pd.read_excel('27514.01.04.2018.09.04.2022.1.0.0.ru.utf8.00000000.xls', skiprows=6)
# Удаляем пропуски
data = data[data['T'].notna()]
# Преобразуем российский формат дат для дальнейшего анализа
data['date'] = pd.to_datetime(data['Местное время в Шереметьево / им. А. С. Пушкина (аэропорт)'], dayfirst=True)
data['dayofyear'] = data['date'].dt.dayofyear
# Вопрос из чата: "как сказать какая будет погода в июле 2022 года, мы можем это сделать?"
# Новый признак: косинус от дня в году
data['cos_dayofyear'] = np.cos((data['dayofyear'] - 1) / 366 * 2 * np.pi)

# Заново переразбиваем датасет на train-test, чтобы изменения применились
data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']

# Из train-test формируем X_train, X_test
X_train = pd.DataFrame()
X_train['cos_dayofyear'] = data_train['cos_dayofyear']  # X
X_test = pd.DataFrame()
X_test['cos_dayofyear'] = data_test['cos_dayofyear']
# "y" оставляем прежним
y_train = data_train['T']
y_test = data_test['T']

# Создаем модель и обучаем ее
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем прогноз
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Печатаем графики
plt.figure(figsize=(20, 5))
plt.scatter(data_train['date'], y_train, label='Data train')
plt.scatter(data_test['date'], y_test, label='Data test')
plt.scatter(data_train['date'], pred_train, label='Predict train')
plt.scatter(data_test['date'], pred_test, label='Predict test')
plt.legend()

# Смотрим на величину ошибки
print('Средняя ошибка на обучающей выборке =', mean_absolute_error(y_train, pred_train))
print('Средняя ошибка на тестовой выборке =', mean_absolute_error(y_test, pred_test))

dates_future = pd.date_range('2022-07-01', '2022-07-31')
data_future = pd.DataFrame()
data_future['date'] = dates_future
data_future['dayofyear'] = data_future['date'].dt.dayofyear
data_future['cos_dayofyear'] = np.cos((data_future['dayofyear'] - 1) / 366 * 2 * np.pi)
X_future = pd.DataFrame()
X_future['cos_dayofyear'] = data_future['cos_dayofyear']
pred_future = model.predict(X_future)

# Печатаем графики
plt.figure(figsize=(20, 5))
plt.scatter(data_train['date'], y_train, label='Data train')
plt.scatter(data_test['date'], y_test, label='Data test')
plt.scatter(data_train['date'], pred_train, label='Predict train')
plt.scatter(data_test['date'], pred_test, label='Predict test')
plt.scatter(data_future['date'], pred_future, label='Predict future')
plt.legend()
plt.show()
print(pred_future)