import pandas as pd
import numpy as np
from multiprocessing import Process, Manager

from const import *
import time

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :start clean data")


# test_ad = pd.read_csv(test_ad_path,na_values='\\N')
# test_click_log = pd.read_csv(test_click_log_path,na_values='\\N')

# train_ad = pd.read_csv(train_ad_path,na_values='\\N')
# train_click_log = pd.read_csv(train_click_log_path,na_values='\\N')
# # train_user = pd.read_csv(train_user_path,na_values='\\N')


# # train_user_id_max = train_click_log.max()

# click_log = pd.concat([train_click_log,test_click_log])
# del train_click_log
# del test_click_log

# ad = pd.concat([train_ad,test_ad])
# del train_ad
# del test_ad

# ad.drop_duplicates( inplace=True)


# ## 0 process data and join ad
# click_log['click_times'] = click_log.click_times.apply(lambda x : min(x,4))
# click_log['weekday'] = click_log.time.apply(lambda x:x%7)
# click_log['weekend'] = click_log.weekday.apply(lambda x:1 if x<=1 else 0)
# click_log = pd.merge(click_log, ad, on='creative_id',how='left')

# del ad

click_log = pd.read_csv(raw_joined)
# click_log.to_csv(raw_joined)

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : raw_joined finished")

# Satistics

####1 user_click_times 

user_click_times = click_log.groupby('user_id')['click_times'].sum()

user_click_times = pd.DataFrame({"user_id":user_click_times.index.values,"total_times":user_click_times.values}).set_index("user_id")

user_click_times['total_times'] = user_click_times['total_times'].fillna(0).astype(np.int16)

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : user_click_times finished")

df = user_click_times




####2 user_id weekend

user_res = []
columns = []
key = 'weekend'
for i in range(click_log[key].min(),click_log[key].max()+1):
    user_weekend_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_weekend_clicks)
    columns.append(key+"_"+str(i))
    del user_weekend_clicks


user_weekend = pd.concat(user_res,axis=1)
user_weekend.columns = columns

for column in columns:
    user_weekend[column] = user_weekend[column].fillna(0).astype(np.int16)

df = df.join(user_weekend)
del user_click_times
del user_weekend

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : user_weekend finished")

####3 user_id/gender/age on weekday
user_res = []
key = 'weekday'
columns = []
for i in range(click_log[key].min(),click_log[key].max()+1):
    user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_clicks)
    columns.append(key+"_"+str(i))
    del user_clicks


user_weekday = pd.concat(user_res,axis=1)
user_weekday.columns = columns

for column in columns:
    user_weekday[column] = user_weekday[column].fillna(0).astype(np.int16)

df = df.join(user_weekday)
del user_weekday

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : user_weekday finished")

####4 user_id/gender/age on product_category
user_res = []
columns = []

key = 'product_category'
for i in range(click_log[key].min(),click_log[key].max()+1):
    user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_clicks)
    columns.append(key+"_"+str(i))
    del user_clicks


user_product_category = pd.concat(user_res,axis=1)
user_product_category.columns = columns

for column in columns:
    user_product_category[column] = user_product_category[column].fillna(0).astype(np.int16)

df = df.join(user_product_category)
del user_product_category

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : product_category finished")

####5 user_id/gender/age on industry
user_res = []
columns = []

key = 'industry'
for i in range(int(click_log[key].min()),int(click_log[key].max())+1):
    user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_clicks)
    columns.append(key+"_"+str(i))
    del user_clicks


user_industry = pd.concat(user_res,axis=1)
user_industry.columns = columns

for column in columns:
    user_industry[column] = user_industry[column].fillna(0).astype(np.int16)

df = df.join(user_industry)
del user_industry

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : industry finished")

### process sequence function

def process_df(user_key,user_ids,user_series):
    i=0
    for user_id, group in user_key.groupby(['user_id']) : #首先对原始数据进行groupby
        user_ids.append(user_id)
        series = list(group.sort_values(by=['time'])[key])
        if len(series)==0:
            series = [1]
        series = '['+','.join(map(str,series))+']'
        user_series.append(series)
        i+=1
        if i%50000==0:
            print(user_id)


## 6 time sequence
max_size = 92
worker = 12
key = 'time'
user_key = click_log[['user_id',key]]
user_key = user_key[user_key[key].notna()]
user_key[key] = user_key[key].apply(lambda x : (x%max_size)).astype(np.int16)

user_key_splited = np.array_split(user_key, worker)

manager =  Manager()
user_ids = manager.list()
user_series = manager.list()

processes =[]
for df in user_key_splited:
    p = Process(target=process_df, args=(df,user_ids,user_series))
    p.start()
    processes.append(p)


for p in processes:
    p.join()

del user_key_splited


user_time = pd.DataFrame({"user_id":list(user_ids),key:list(user_series)}).set_index("user_id")

del user_ids 
del user_series

df = df.join(user_time)
del user_time

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : time finished")

## 7 advertiser_id sequence
max_size = 26713
worker = 12
key = 'advertiser_id'
user_key = click_log[['user_id','time',key]]
user_key = user_key[user_key[key].notna()]
user_key[key] = user_key[key].apply(lambda x : (x%max_size)+1).astype(np.int16)
user_key_splited = np.array_split(user_key, worker)

manager =  Manager()
user_ids = manager.list()
user_series = manager.list()

processes =[]
for df in user_key_splited:
    p = Process(target=process_df, args=(df,user_ids,user_series))
    p.start()
    processes.append(p)


for p in processes:
    p.join()

del user_key_splited


user_advertiser_ids = pd.DataFrame({"user_id":list(user_ids),key:list(user_series)}).set_index("user_id")


del user_ids 
del user_series


df = df.join(user_advertiser_ids)
del user_advertiser_ids


print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : advertiser_id finished")

## 8 product id sequence
max_size = 26713
worker = 12
key = 'product_id'
user_key = click_log[['user_id','time',key]]
user_key = user_key[user_key[key].notna()]
user_key[key] = user_key[key].apply(lambda x : (x%max_size)+1).astype(np.int16)
user_key_splited = np.array_split(user_key, worker)

manager =  Manager()
user_ids = manager.list()
user_series = manager.list()

processes =[]
for df in user_key_splited:
    p = Process(target=process_df, args=(df,user_ids,user_series))
    p.start()
    processes.append(p)


for p in processes:
    p.join()

del user_key_splited


user_product_id = pd.DataFrame({"user_id":list(user_ids),key:list(user_series)}).set_index("user_id")


del user_ids 
del user_series

df = df.join(user_product_id)
del user_product_id

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : product_id finished")

df.to_csv(raw_processed)

print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  : raw_processed finished")


# df.total_times.quantile([0.2,0.5,0.75,0.9,0.95,0.98,0.99])
# 0.20     28.0
# 0.50     48.0
# 0.75     83.0
# 0.90    137.0
# 0.95    187.0
# 0.98    266.0
# 0.99    335.0

########################  for dnn input
# 1. filter total times

# df = df[df.total_times<=335]

# df = df[df.user_id<=900000]

# # 2.1 0.9 na threshold 

# columns = ['total_times','time','advertiser_id','weekend_0','product_id','weekend_1','product_category_18','weekday_1','weekday_0','weekday_4','weekday_3','weekday_2','weekday_5','weekday_6','industry_6','product_category_2','product_category_5','industry_319','industry_322','industry_247','industry_54','industry_317','product_category_3','industry_297','industry_238','industry_242','industry_73','product_category_12','industry_88','industry_289','product_category_8','industry_60','industry_248','industry_25','product_category_17','industry_326','industry_246','industry_21','industry_291','industry_5','industry_318','industry_47','industry_296','industry_329','industry_36','industry_40','industry_252','industry_27','industry_26','industry_183','industry_203','industry_202','industry_253','product_category_13','industry_321','industry_288','industry_259','industry_205']

# new_df = df[columns]


# ## 6  :creative_id,ad_id,product_id,advertiser_id
# # 6.1 creative_id
# max_size = 7417

# user_res = []
# columns = []

# term = 'creative_id'
# click_log[term] = click_log[term].apply(lambda x : x % max_size)
# for i in range(max_size):
#     user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
#     user_res.append(user_clicks)
#     columns.append(key+"_"+str(i))


# user_creative_id = pd.concat(user_res,axis=1)
# user_creative_id.columns = columns

# # 6.2 ad_id
# max_size = 7417

# user_res = []
# columns = []

# term = 'ad_id'
# click_log[term] = click_log[term].apply(lambda x : x % max_size)
# for i in range(max_size):
#     user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
#     user_res.append(user_clicks)
#     columns.append(key+"_"+str(i))


# user_ad_id = pd.concat(user_res,axis=1)
# user_ad_id.columns = columns


# # 6.3 product_id
# max_size = 7417

# user_res = []
# columns = []

# term = 'product_id'
# click_log[term] = click_log[term].apply(lambda x : x % max_size)
# for i in range(max_size):
#     user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
#     user_res.append(user_clicks)
#     columns.append(key+"_"+str(i))


# user_product_id = pd.concat(user_res,axis=1)
# user_product_id.columns = columns


# # 6.3 advertiser_id
# max_size = 7417

# user_res = []
# columns = []

# term = 'advertiser_id'
# click_log[term] = click_log[term].apply(lambda x : x % max_size)
# for i in range(max_size):
#     user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
#     user_res.append(user_clicks)
#     columns.append(key+"_"+str(i))


# user_advertiser_id = pd.concat(user_res,axis=1)
# user_advertiser_id.columns = columns


# df = pd.join([user_click_times,user_weekend,user_weekday,user_time,user_industry,user_product_category,user_creative_id,
#     user_ad_id,user_advertiser_id,user_product_id])