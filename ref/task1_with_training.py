




import params
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
import models
from keras.callbacks import ModelCheckpoint
from joblib import dump


# Читаем оригинальные файлы
sales = pd.read_csv(params.sales, delimiter=params.delimeter)
customers = pd.read_csv(params.customers, delimiter=params.delimeter)

"""
Данные по клиентам отстутсвуют для village. Сперва я решил заполнить их с помощью функции
из utils, где я беру данные из других городов и уменшьаю количество поенциальных клиентов. Но потом я решил, что
это совсем уж гадания на кофейной гуще и попросту решил удалить все записи из village. Да, это плохо
"""
sales = utils.remove_vl(sales)
sales = sales.reset_index(drop=True)

"""
В колонке price датасета sales есть NaN. Вместо их удаления я решил заполнить их средними значениями цены
этих СКУ за предыдущие годы и другие локации. Стоит ли так делать - это вопрос,
но для простоты я решил сделать именно так.
"""
utils.find_and_replace_price_nans(sales)
print(sales.head)

"""
Чтобы датасет по клиентам мог использоваться для анализа, я меняю его формат так,
чтобы его можно было left join-ить с датасетом sales
"""
modified_customers = utils.create_customer_features(customers)
print(modified_customers.head)


"""
Делаем один датасет из sales и customers
"""
cols = ['year','month', 'ira']
new_dataset = utils.merge_datasets(sales, modified_customers, cols)


"""
Теперь разберемся с categorical vars
"""
# IRA и location дают одну и ту же информацию, удаляю location
new_dataset.drop(['location'], axis=1, inplace=True)

# one-hot encoded categorical variables. Чтобы избежать dummy variables trap ставить drop_first=True
cols_to_transform = ['sku_id', 'ira', 'product_category', 'brand', 'shape', 'filling']
new_dataset = pd.get_dummies(new_dataset, columns = cols_to_transform, drop_first=True )

# Конвертируем alcohol variable в boolean 0 or 1
new_dataset['with_alcohol'] = new_dataset['with_alcohol'].replace({'Yes': True, 'No': False})*1

# Сохраним этот датасет
new_dataset.to_pickle("data/new_dataset_no_categorical_no_vl.pkl")
print("The dataset was saved")


"""
Теперь у нас есть датасет. Если запускаем второй раз, то все выше можно закомментить и сделать это:
    
with open('data/new_dataset_no_categorical_no_vl.pkl', 'rb') as f:
    data = pickle.load(f) 
    
    
Теперь можно перейти к моделям
"""

# Подготовим датасет для модели
y = new_dataset.volume
X = new_dataset.drop(["volume"], axis=1)
y = y.to_numpy()
X = X.to_numpy()

# Разделяем на трэйн и тест. k-folds использовать не буду, идем по простому пути
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

"""
ОПЦИЯ 1 - Random Forest
Буду использовать именно ее в итоге. 
GridSearch не использую, просто тупо беру модель
SMAPE: ~24

Другие модели можно раскоментить
"""
rf = models.get_rf_model()

# Тренируем модель
rf.fit(X_train, y_train)

# Сохраняем модель
dump(rf, 'models/rf_model.joblib') 

# Делаем прогноз. Расстраиваемся, что результат так себе, но что поделать, времени мало
y_pred = rf.predict(X_test)
y_pred = y_pred.astype(int)
print('SMAPE FOR RF: {}'.format(utils.smape(y_test, y_pred)))


"""
ОПЦИЯ 2 - Lasso.
LR сильно overfit-ит, лассо может это исправить
SMAPE: ~28


lasso_pipe = models.get_lasso_model()
lasso_pipe.fit(X_train, y_train)

dump(rf, 'models/lasso_model.joblib') 

# Это не метрика smape, так что можно не обращать внимание
print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test, y_test)))

y_pred = lasso_pipe.predict(X_test)
print('SMAPE FOR LASSO: {}'.format(utils.smape(y_test, y_pred)))
"""

"""
ОПЦИЯ 3 - Neural Network
SMAPE: ~23.8


from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=models.get_nn_model, batch_size=32,epochs=400)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y = y.reshape(-1,1)

checkpoint = ModelCheckpoint('models/nn_model.h5', 
                             monitor='val_loss', 
                             verbose=2, 
                             save_best_only=True, 
                             mode='min')

results=regressor.fit(X_train,y_train, callbacks=[checkpoint])

y_pred= regressor.predict(X_test)
print('SMAPE FOR NN: {}'.format(utils.smape(y_test, y_pred)))
"""



