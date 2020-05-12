import pandas as pd
import numpy as np
from multiprocessing import Process, Manager

from const import *


# test_ad = pd.read_csv(test_ad_path,na_values='\\N')
# test_click_log = pd.read_csv(test_click_log_path,na_values='\\N')

# train_ad = pd.read_csv(train_ad_path,na_values='\\N')
# train_click_log = pd.read_csv(train_click_log_path,na_values='\\N')
# train_user = pd.read_csv(train_user_path,na_values='\\N')


# train_user_id_max = train_click_log.max()

# click_log = pd.concat([train_click_log,test_click_log])

# ad = pd.concat([train_ad,test_ad])

# ##process click_log added time attribuate
# click_log['click_times'] = click_log.click_times.apply(lambda x : min(x,4))
# click_log['weekday'] = click_log.time.apply(lambda x:x%7)
# click_log['weekend'] = click_log.weekday.apply(lambda x:1 if x<=1 else 0)
# click_log = pd.merge(click_log, ad, on='creative_id',how='left')


# user_ids = click_log[['user_id']]

click_log = pd.read_csv(raw_joined)

user_click_times = click_log.groupby('user_id')['click_times'].sum()

# Satistics

####1 user_id time
user_res = []
columns = []
key = 'time'
for i in range(click_log[key].min(),click_log[key].max()+1):
    user_weekend_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_weekend_clicks)
    columns.append(key+"_"+str(i))


user_time = pd.concat(user_res,axis=1)
user_time.columns = columns


####2 user_id weekend

user_res = []
columns = []
key = 'weekend'
for i in range(click_log[key].min(),click_log[key].max()+1):
    user_weekend_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_weekend_clicks)
    columns.append(key+"_"+str(i))


user_weekend = pd.concat(user_res,axis=1)
user_weekend.columns = columns

####3 user_id/gender/age on weekday
user_res = []
key = 'weekday'
columns = []
for i in range(click_log[key].min(),click_log[key].max()+1):
    user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_clicks)
    columns.append(key+"_"+str(i))


user_weekday = pd.concat(user_res,axis=1)
user_weekday.columns = columns


####4 user_id/gender/age on product_category
user_res = []
columns = []

key = 'product_category'
for i in range(click_log[key].min(),click_log[key].max()+1):
    user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_clicks)
    columns.append(key+"_"+str(i))


user_product_category = pd.concat(user_res,axis=1)
user_product_category.columns = columns


####5 user_id/gender/age on industry
user_res = []
columns = []

key = 'industry'
for i in range(int(click_log[key].min()),int(click_log[key].max())+1):
    user_clicks = click_log[click_log[key]==i].groupby('user_id')['click_times'].sum()
    user_res.append(user_clicks)
    columns.append(key+"_"+str(i))


user_industry = pd.concat(user_res,axis=1)
user_industry.columns = columns


def process_df(user_key,user_ids,user_series):
    i=0
    for user_id, group in user_key.groupby(['user_id']):#首先对原始数据进行groupby
        user_ids.append(user_id)
        user_series.append(list(group.sort_values(by=['time'])[key]))
        i+=1
        if i%50000==0:
            print(user_id)


## 6 advertiser_id sequence
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

user_advertiser_ids = pd.DataFrame({"index":user_ids,key:user_series})



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