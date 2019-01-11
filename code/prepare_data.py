import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

args = sys.argv
order_file_name = args[1]
group_file_name = args[2]
airline_file_name = args[3]
train_set_file_name = args[4]
total_dataset = args[5]


order = pd.read_csv(order_file_name)
group = pd.read_csv(group_file_name)
airline = pd.read_csv(airline_file_name)

group['product_name_price_min'] = group['price'].groupby(group['product_name']).transform('min')
merge_data = order.merge(group, on=['group_id'], how='left')

#cp值:價錢/天數
merge_data["cp"] = merge_data["price"] / merge_data['days']

#source
source_1_dummy = pd.get_dummies(merge_data["source_1"] )
source_2_dummy = pd.get_dummies(merge_data["source_2"] )
merge_data = pd.concat([merge_data , source_1_dummy] , axis = 1)
merge_data = pd.concat([merge_data , source_2_dummy] , axis = 1)



#同一行程是否為最低價
merge_data['price-min'] = merge_data['price'] - merge_data['product_name_price_min']
merge_data['is-min-price'] = 0
merge_data.loc[merge_data['price-min'] == 0 , ['is-min-price']] = 1

#同一行程有有多少訂單
merge_data['num_same_group'] = merge_data[['order_id']].groupby(merge_data['group_id']).transform('count')


#同意行程總共多少人
merge_data['total_people_amount'] = merge_data[['people_amount']].groupby(merge_data['group_id']).transform('sum')


#行程 優惠
merge_data["discount"] = 0
merge_data.loc[merge_data["product_name"].str.contains("省") == True , ["discount"]] = 1
merge_data.loc[merge_data["product_name"].str.contains("折") == True , ["discount"]] = 1
merge_data.loc[merge_data["product_name"].str.contains("贈") == True , ["discount"]] = 1
merge_data.loc[merge_data["product_name"].str.contains("送") == True , ["discount"]] = 1
merge_data.loc[merge_data["product_name"].str.contains("減") == True , ["discount"]] = 1
merge_data.loc[merge_data["product_name"].str.contains("優惠") == True , ["discount"]] = 1
merge_data.drop(['product_name'], axis=1, inplace=True)


#時間格式轉換，以及時間處理
merge_data['begin_date'] = pd.to_datetime(merge_data['begin_date'])
merge_data['order_date'] = pd.to_datetime(merge_data['order_date'])
merge_data['begin_date_month'] = merge_data["begin_date"].dt.month
merge_data['order_date_month'] = merge_data["order_date"].dt.month
merge_data['order_date_dayofweek'] = merge_data['order_date'].dt.dayofweek
merge_data['begin_date_dayofweek'] = merge_data['begin_date'].dt.dayofweek
merge_data['order_date_isweekend'] = 0
merge_data['begin_date_isweekend'] = 0
merge_data.loc[merge_data['order_date_dayofweek'] == 5  , ['order_date_isweekend']] = 1
merge_data.loc[merge_data['order_date_dayofweek'] == 6  , ['order_date_isweekend']] = 1
merge_data.loc[merge_data['begin_date_dayofweek'] == 5  , ['order_date_isweekend']] = 1
merge_data.loc[merge_data['begin_date_dayofweek'] == 6  , ['order_date_isweekend']] = 1




# 航班處理
#去程起飛時間，回程抵達時間
go_fly = airline[["group_id" , "fly_time" , "arrive_time"]]
go_fly['fly_time'] = airline['fly_time'].groupby(airline['group_id']).transform('min')
go_fly['fly_time'] = pd.to_datetime(go_fly['fly_time'])
go_fly['arrive_time'] = airline['arrive_time'].groupby(airline['group_id']).transform('max')
go_fly['arrive_time'] = pd.to_datetime(go_fly['arrive_time'])
go_fly = go_fly.drop_duplicates()
merge_data = merge_data.merge(go_fly, on=['group_id'], how='left')

#整個行程搭了幾次飛機
count = airline.groupby(['group_id']).size().to_frame("fly_count")
merge_data = merge_data.merge(count, on=['group_id'], how='left')

#刪除沒用到的欄位
merge_data.drop(['source_1'], axis=1, inplace=True)
merge_data.drop(['source_2'], axis=1, inplace=True)
merge_data.drop(['unit'], axis=1, inplace=True)
merge_data.drop(['area'], axis=1, inplace=True)
merge_data.drop(['sub_line'], axis=1, inplace=True)
merge_data.drop(['promotion_prog'], axis=1, inplace=True)


training_set = pd.read_csv(train_set_file_name)
merge_data = merge_data.merge(training_set , on=['order_id'], how='left')
merge_data = merge_data.dropna()   #刪除有缺值的列

print(merge_data.info())
merge_data.to_csv(total_dataset , index = False)

 