import pandas  as pd
import numpy as np
import math

project = '/home/tione/notebook/'

click_log_path = project +'train/click_log_joined.csv'
click_log = pd.read_csv(click_log_path)

def rmnan(x):
    if not x or len(x)==0:
        return x
    return x.replace(' nan,','').replace(', nan','')

#np.int16 (-32768 to 32767）

## 1.process train file
user_time_path = project + 'train/user/user_time.csv'
user_creative_id_path = project + 'train/user/user_creative_id.csv'
user_ad_id_path = project + 'train/user/user_ad_id.csv'
user_advertiser_id_path = project + 'train/user/user_advertiser_id.csv'
user_industry_path = project + 'train/user/user_industry.csv'
user_product_id_path = project + 'train/user/user_product_id.csv'
user_product_category_path = project + 'train/user/user_product_category.csv'


#1.1 time  91 91
k = 92
term = 'time'
user_time = pd.read_csv(user_time_path)
user_time[term] = user_time[term].apply(lambda x : list(map(lambda y : int(y%k),list(filter(lambda x:not math.isnan(x),eval(rmnan(str(x))))))))
# user_time[term]= user_time[term].astype(np.int16)
user_time = user_time[user_time['user_id']!=839367]

#1.2 creative_id 2481135  4445718
k = 26713
term = 'creative_id'
user_creative_id = pd.read_csv(user_creative_id_path)
user_creative_id[term] = user_creative_id[term].apply(lambda x : list(map(lambda y : int(y%k),list(filter(lambda x:not math.isnan(x),eval(rmnan(str(x))))))))
# user_creative_id[term] = user_creative_id[term].astype(np.int16)

#1.3 ad_id 2264190 3812200
k = 26713
term = 'ad_id'
user_ad_id = pd.read_csv(user_ad_id_path)
user_ad_id[term] = user_ad_id[term].apply(lambda x : list(map(lambda y : int(y%k),list(filter(lambda x:not math.isnan(x),eval(rmnan(str(x))))))))
# user_ad_id[term] = user_ad_id[term].astype(np.int16)


#1.4 advertiser_id  52090 62965.0
k = 26713
term = 'advertiser_id'
user_advertiser_id = pd.read_csv(user_advertiser_id_path)
user_advertiser_id[term] = user_advertiser_id[term].apply(lambda x : list(map(lambda y : int(y%k),list(filter(lambda x:not math.isnan(x),eval(rmnan(str(x))))))))
# user_advertiser_id[term] = user_advertiser_id[term].astype(np.int16)

#1.5 industry 326 335.0
k = 336
term = 'industry'
user_industry = pd.read_csv(user_industry_path)
user_industry[term] = user_industry[term].apply(lambda x : list(map(lambda y : int(y%k),list(filter(lambda x:not math.isnan(x),eval(rmnan(str(x))))))))
# user_industry[term] = user_industry[term].astype(np.int16)

#1.6 product_id 33273 44313.0
k = 26713
term = 'product_id'
user_product_id = pd.read_csv(user_product_id_path)
user_product_id[term] = user_product_id[term].apply(lambda x : list(map(lambda y : int(y%k),list(filter(lambda x:not math.isnan(x),eval(rmnan(str(x))))))))
# user_product_id[term] = user_product_id[term].astype(np.int16)

#1.7 product_category  18 18
k = 19
term = 'product_category'
user_product_category = pd.read_csv(user_product_category_path)
user_product_category[term] = user_product_category[term].apply(lambda x : list(map(lambda y : int(y%k),list(filter(lambda x:not math.isnan(x),eval(rmnan(str(x))))))))
# user_product_category[term] = user_product_category[term].astype(np.int16)

df = pd.concat([user_time,user_creative_id,user_ad_id,user_advertiser_id,user_industry,user_product_id,user_product_category],axis=1)


