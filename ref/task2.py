
import params
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.formula.api import ols 


# Читаем оригинальный файл с продажами
sales = pd.read_csv(params.sales, delimiter=params.delimeter)

# Читаем файл с прогнозами из первого задания
predicted = pd.read_csv('results/results_for_2nd_task.tsv', delimiter=params.delimeter)

# Заменяем NaN в колонке "Цена" на средние значения этого товара
utils.find_and_replace_price_nans(sales)

# Убираем ненужные колонки
sales.drop(['ira', 'product_category', 'item_per_bundle', 'shape', 'with_alcohol', 'filling'],
           axis=1, 
           inplace=True)

# Создаем один датасет из оригинального и того, где прогнозы
df = pd.concat([sales, predicted])

# Сортируем по году и месяцу
df.sort_values(by=['year', 'month'], inplace=True)

# Получаем список всех брендов
brands = df.brand.unique().tolist()

# Я решил разбить датасет на несколько датасетов с соответствуюшими брендами
# Таким образом цена будет предсказываться на SKU/бренд

datasets = {}
for brand in brands:
    brand_df  = df[df.brand == brand]
    datasets[brand] = brand_df
    print(brand_df.head)


"""
Анализ получился никаким по большому счету, ибо мне нужно больше времени разбираться что там как и почему
происходит. К тому же у большинства СКУ зависимость цена-продажи нелинейная, и мне нужно искать подходящий
метод нелинейной оптимизации, а например в scikit похоже нельзя получить
саммари модели как в statsmodels, а в статсмоделс я ничего нелинейного пока не нашел. В общем, тут надо 
углубляться в эту тему, я сейчас этого делать не буду.
Но в целом, то что внизу, показывает в каком направлении тут можно мыслить.

"""

skus = df.sku_id.unique().tolist()
skus_original_prices = []
skus_forecasted_prices = []

# соответствующие бренды и категории продуктов
# можно было добавить и остальные колонки, например категории и алкоголь и потом проанализировать более детально
# но уже нет времени. Плюс я немного протупил и уже удалил их.
# Наверняка там бы проявились какие-то зависимости, может по категориям
skus_brands = []

# Пока сделаю для каждого города отдельно и для месяца
# Здесь просто задаем локацию и нужный месяц
# Файл будет сохранен в папке results с именем локации и номером месяца
chosen_location = 'Alpha City'
month = 9

for sku in skus:
    # Берем бренд для этого СКУ
    active_brand = df.loc[(df.sku_id == sku), 'brand'].values[0]
    
    
    # Выбираем датасет для этого бренда
    brand_df = datasets[active_brand]
    
    # Оставляем данные только для выбранного СКУ
    brand_df = brand_df[brand_df.sku_id == sku]
    
    # Берем все цены для этого СКУ и получаем среднее. Это мне нужно, чтобы от этой цифры отталкиваться
    # при варьировании цены плюс минус 15%
    all_prices_for_sku_id = brand_df.loc[(brand_df.month == month) & 
                                         (brand_df.year == 2019) &
                                         (brand_df.location == chosen_location), 'price']  
     
    original_price = all_prices_for_sku_id.values[0]
    
    # Добавим среднюю цену для СКУ в список
    skus_original_prices.append(original_price)
    
    # Добавим бренд и категорию. Немного тупо, но я это делаю уже после всего остального на самом деле.
    skus_brands.append(active_brand)
    
    # Создаем список цен - потенциаьных кандидатов
    # от -15 до +15 процентов, с шагом в 5 процентов (почему бы и не 5)     
    lower_bound = original_price - original_price*0.15
    upper_bound = original_price + original_price*0.15
    step = original_price*0.05 # 5% step
    prices = [x for x in np.arange(lower_bound, upper_bound, step)]
    
    # Строим график зависимости цены от объема, чтобы лишний раз расстроиться
    if params.show_info:
        sns.lmplot(x="price", y="volume", data=brand_df, fit_reg=True, height=4)
        plt.show()
    
    """
    Берем OLS модель и выводим саммари, саммари нужна для того, чтобы посмотреть коэффициенты, которые нужны
    для уравнений, которые, как я посмотрел, используются для оптимизации выручки
        profit = revenue - cost
        revenue = demanded_volume * price
        profit = demanded_volume * price - cost
    
    Предположим, что cost константа, считаем, что demanded_volume = f(price) - график построили выше
    Допустим зависимость не полная хрень, и я решил, что я могу назвать ее линейной
    Тогда demanded_volume = coef1+coef2*price
    Из саммари я беру эти коэффициенты и задаю их ниже
    В саммари смотрим на Intercept (coef1) и price (coef2)
    """
    model = ols("volume ~ price", data=brand_df).fit() 
    results_summary = model.summary()
    if params.show_info:
        print(results_summary)
    
    # Основываясь ни на чем, я сделал cost 20 процентов от средней цены СКУ
    cost = original_price * 0.2
    # Вообще я могу обойтись полностью без этого значения, но ладно уж, пусть будет
    # Чтобы не путаться лишний раз в понятиях выручки, прибыли дохода и так далее. Пусть будет так как есть
    
    revenue = [] 
    
    # Коэффициенты, полученные из саммари модели
    coef1, coef2 = utils.get_coeffs_from_summary(model)
    
    # Получаем значения revenue для каждой цены
    for price in prices:     
        volume_demanded=coef1+coef2*price    
        revenue.append((price-cost)*volume_demanded) # фукнция профита  
        
    
    # Создаем датафрейм
    profit = pd.DataFrame({"price": prices, "revenue": revenue}) 
    
    # Строим график Цена-Выручка
    if params.show_info:
        plt.plot(profit["price"], profit["revenue"])
    
    # Цена при которой выручка будет максимальна для данного СКУ
    val = profit[profit['revenue'] == profit['revenue'].max()]
    price = val['price'].values[0]
    revenue = val['revenue'].values[0]
    
    # Добавим выбранную цену для СКУ в список
    skus_forecasted_prices.append(price)
    
    print("Max price/revenue: {} / {}".format(price, revenue))
    

"""
Теперь создадим датасет и посмотрим на сколько для каждого СКУ модель предложила изменить цену. 
Держа в уме, что модель не катит все-таки
"""

resulting_df = pd.DataFrame()
resulting_df['sku_id'] = skus

# можно еще добавить колонки с брендами. чисто для визуального анализа
# времени более детально анализировать уже нет, к сожалению
resulting_df['brand'] = skus_brands

resulting_df['original_price'] = skus_original_prices
resulting_df['forecasted_price'] = skus_forecasted_prices


# Считаем разницу в ПРОЦЕНТАХ
resulting_df['difference'] = resulting_df.apply(lambda x: utils.get_change(x['original_price'], x['forecasted_price']), axis=1)

resulting_df.to_pickle("results/price_difference_{}_{}.pkl".format(chosen_location, month))
