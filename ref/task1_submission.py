
import params
import utils
import pandas as pd
import numpy as np
from joblib import load

"""
В этом файле много повторов с файлом task1_with_training. Немножко намудрил тут
с созданием датасета для итогового файла результатов. Плюс нужен датасет для второго задания.
В общем, мало времени остается до дедлайна, поэтому просто повторю некоторый код тут
"""

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
modified_customers = utils.create_customer_features(customers)

cols = ['year','month', 'ira']
new_dataset = utils.merge_datasets(sales, modified_customers, cols)


"""
Создаем датасет на 3 месяца в 2019 году, на все локации и СКУ, этот датасет будет подан в модель для прогноза.
Использую функцию из utils - create_submission_dataframe, внутри функции тоже описаны шаги.
"""
year = 2019
months = [7, 8, 9]
iras = ['AC', 'MN', 'PB']
skus = sales.sku_id.unique().tolist() # получаем список СКУ
res_df = utils.create_submission_dataframe(sales, year, months, iras, skus)
print("Submission dataset was successfully created")
print(res_df.head)


# Добавляем к созданному датасету данные о клиентах из таблицы customers. Соотносим по году, месяцу и ира
cols = ['year','month', 'ira']
new_dataset = utils.merge_datasets(res_df, modified_customers, cols)
print("Merged dataset was successfully created")
print(new_dataset.head)

            
# Сохраним датасет на будущее на всякий случай, хотя он не нужен, 
# ниже я сохранию датасет без categorical vars
new_dataset.to_pickle("data/new_dataset_for_submission_no_vl.pkl")

"""
Теперь разберемся с categorical vars
"""

# one-hot encoded categorical variables. Чтобы избежать dummy variables trap ставить drop_first=True
cols_to_transform = ['sku_id', 'ira', 'product_category', 'brand', 'shape', 'filling']
new_dataset = pd.get_dummies(new_dataset, columns = cols_to_transform, drop_first=True )

# Конвертируем alcohol variable в boolean 0 or 1
new_dataset['with_alcohol'] = new_dataset['with_alcohol'].replace({'Yes': True, 'No': False})*1

# Сохраним этот датасет
new_dataset.to_pickle("data/new_dataset_for_submission_no_categorical_no_vl.pkl")


"""
Теперь используем натренированную модель Random Forest для прогноза
"""
rf = load('models/rf_model.joblib') 
y_pred = rf.predict(new_dataset) # прогноз volume
y_pred = y_pred.astype(int) # флоаты нам не нужны


# Подготовим датасет для записи в файл с результатами
res_df.drop(['product_category', 'item_per_bundle', 'shape', 'with_alcohol', 'filling'], axis = 1, inplace=True) 
res_df['volume'] = y_pred # добавим колонку с прогнозом
# В задании хотите location вместо ira - так и сделаю)
res_df['ira'] = np.where(res_df['ira'] == 'AC', 'Alpha City', res_df['ira'])
res_df['ira'] = np.where(res_df['ira'] == 'MN', 'Moon', res_df['ira'])
res_df['ira'] = np.where(res_df['ira'] == 'PB', 'Pirate Bay', res_df['ira'])
res_df.rename(columns={"ira": "location"}, inplace=True)


# Сохраню файл отдельно для второго задания, ибо мне там пару колонок пригодится в отличии от файла 
# результатов для Задания 1 
res_df.to_csv('results/results_for_2nd_task.tsv', index=None, header=True, sep = params.delimeter)

# Теперь убираю ненужные колонки и сохраняю в файл со скудными результатами
res_df.drop(['price', 'brand'], axis=1, inplace=True)
res_df.sort_values(by=['month','sku_id'], inplace=True)
res_df.to_csv('results/results.tsv', index=None, header=True, sep = params.delimeter)

            