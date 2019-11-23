
import pandas as pd
import params
import numpy as np
import random


def find_and_replace_price_nans(sales_data):
    """
    В колонке price есть NaN. Функция заменяет их средними значениями для данного товара
    """
    sku_averages = {}
    
    # Ищем строки где есть НаН
    rows = sales_data.loc[pd.isna(sales_data["price"]), :].index.tolist()
    # Получаем СКУ у которых нет цены
    list_of_skus_without_price =  [sales_data['sku_id'].iloc[row_index] for row_index in rows]
    # Удаляем дубликаты
    list_of_skus_without_price = list(set(list_of_skus_without_price))
    
    # Находим среднее значения для каждого СКУ из других строк, это значение будет поставлено на место НаН
    for sku in list_of_skus_without_price:
        # find all the prices for that sku
        all_prices_for_sku_id = sales_data.loc[(sales_data.sku_id == sku) & (sales_data.price.notnull()), 'price']       
        if all_prices_for_sku_id.any():
            sku_mean = all_prices_for_sku_id.mean()
            sku_averages[sku] = sku_mean
        else:
            print("This sku has literally no price")     
            
    print("Average values for sku with NaNs were found")     
  
    for row in rows:   
        sku = sales_data.get_value(row,'sku_id')
        sales_data.at[row, 'price'] = sku_averages[sku]
        if params.test_mode:
            print('Replaced {} row for sku {} with {} average value'.format(row, sku, sku_averages[sku]))



def create_customer_features(customers_data):

    """
    Функция создает новый датасет, в формате, подходящем для merge с датасетом sales.
    На каждую строку в sales, будет добавлена информация о клиентах из датасета customers.
    """
    
    resulting_df = pd.DataFrame()
    window = 20 # по количеству типов клиентов
    start = 0
    end = start + window
    
    while end <= customers_data.shape[0]:
        
        indexes = np.arange(start, end)
        df = pd.DataFrame()
        for index in indexes:     
            year = customers_data.get_value(index,'year')
            month = customers_data.get_value(index,'month')
            ira = customers_data.get_value(index,'ira')
            df.at[0, 'year'] = year
            df.at[0, 'month'] = month
            df.at[0, 'ira'] = ira
            customer_type = customers_data.get_value(index,'customers_type')
            customer_amount = customers_data.get_value(index,'amount_of_customers')
            
            df.at[0, customer_type] = customer_amount
            
        resulting_df = pd.concat([resulting_df, df])
        start = end
        end = end + window
    
    return resulting_df


def deal_with_village(df):
    """
    Функция заполняет значения для Village беря данные из Pirate Bay и деля их на число от 9 до 14
    Не использую эту функцию.
    """
    
    rows = df.loc[pd.isna(df["Archer"]), :].index.tolist()
       
    # gotta deal with fckn VL
    types = ['Archer', 
             'Monk', 
             'Dwarf', 
             'Centaur', 
             'Gremlin', 
             'Genie', 
             'Demon', 
             'Ghost', 
             'Minotaur', 
             'Medusa',
             'Goblin',
             'Orc',
             'Gargoyle',
             'Pegasus',
             'Pirate',
             'Unicorn',
             'Giant',
             'Elf',
             'Angel',
             'other']
    
    index_in_pb = 1724 # just an index of the row, with info about PB city customers

    for col_type in types:
        replacement_value = random.randint(9, 15) # same for all rows, but should differ a bit for columns
        for row in rows:
            df.at[row, col_type] = int(df.at[index_in_pb, col_type] / replacement_value)
                        
    # check if there are NaNs left
    if df.loc[pd.isna(df["Archer"]), :].index.tolist() == []:
        print("All NaNs were replaced")
            



def merge_datasets(df1, df2, cols, how='left'):
    """
    Left join двух датасетов
    """
    return pd.merge(left=df1, right=df2, on=cols, how=how)

def remove_vl(df):
    """
    Удалить все записи из Village
    """
    df = df[df.ira != 'VL']
    return df


def create_submission_dataframe(sales, year, months, iras, skus):
    """
    Функция создает датасет для предсказывания продаж, для кажлого СКУ, локации, года и месяца
    
    :param sales: Оригинальный датасет
    :param year: Год
    :param months: Список месяцев
    :param iras: Список локаций
    :param skus: Список СКУ
    
    """
    
    # Сюда будет добавляться каждая новая генерированная строка
    resulting_df = pd.DataFrame()
      
    # Я не в восторге от тройного лупа, но делаю быстро. 
    for month in months:
        for ira in iras:
            for sku in skus:
                df = pd.DataFrame()
                
                indexes = sales.loc[(sales.sku_id == sku), 'price'].index.to_list()
                ind = indexes[0]
                
                df.at[0, 'year'] = year
                df.at[0, 'month'] = month
                df.at[0, 'ira'] = ira
                
                df.at[0, 'product_category'] = sales.get_value(ind, 'product_category')
                df.at[0, 'brand'] = sales.get_value(ind, 'brand')
                df.at[0, 'sku_id'] = sku
                df.at[0, 'item_per_bundle'] = sales.get_value(ind, 'item_per_bundle')
                df.at[0, 'shape'] = sales.get_value(ind, 'shape')
                df.at[0, 'with_alcohol'] = sales.get_value(ind, 'with_alcohol')
                df.at[0, 'filling'] = sales.get_value(ind, 'filling')
                
                """
                Цену надо брать за июнь 2019. Не везде она есть. Иногда ее нет в этой локации. Сначала
                хотел удалить эти строки, но таких довольно много, поэтому я решил сделать проще 
                (про правильнольность это другой вопрос) и сперва разрешить брать цену из других годов, 
                а потом и вовсе заменить ценой, например, из другого города с помощью функции try_to_find_another_price
                       
                """
                price = sales.loc[((sales.year == 2019) | (sales.year == 2018) | (sales.year == 2017)) & 
                                  (sales.month == 6) &
                                  (sales.sku_id == sku) &
                                  (sales.ira == ira), 'price'] 
                
                if price.any():
                    df.at[0, 'price'] = price.values[0]
                else:
                    if params.test_mode:
                        print('Price of {} is not available in {} location'.format(sku, ira))
                    # Попробуем найти цену из другой локации или месяца, короче любую
                    price = try_to_find_another_price(sales, sku)
                    df.at[0, 'price'] = price.values[0]
                
                # Добавляем строку в итоговый датасет
                resulting_df = pd.concat([resulting_df, df])
            
    return resulting_df


def try_to_find_another_price(sales, sku):
    """
    Для некоторых локации нет цены в последнем месяце. Чтобы не удалять эти строки, я заменяю эти цены
    другой ценой для этого СКУ
    """
    
    price = sales.loc[((sales.year == 2019) | (sales.year == 2018) | (sales.year == 2017)) & 
                                  (sales.sku_id == sku), 'price'] 
    return price


def smape(A, F):
    """
    SMAPE метрика, подходящая для numpy array
    """
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def get_coeffs_from_summary(results):
    """
    Достает intercept и price параметры из саммари
    """
    coeff = results.params
    return coeff['Intercept'], coeff['price']


def get_change(current, previous):
    """
    Функция дает изменение цены в процентах
    """
    
    if current == previous:
        return 0
    try:
        return ((current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0
