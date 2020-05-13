import pandas  as pd
import numpy as np
import math

from const import *
from utils import *

#np.int16 (-32768 to 32767）

click_log_test = pd.read_csv(click_log_test_path)



## 1.process test file

#1.1 time  91 91
k = 92
term = 'time'
user_time = pd.read_csv(user_time_test_path)
user_time[term] = user_time[term].apply(lambda x : list(map(lambda y : int(y%k),eval(str(x)))))
# user_time[term]= user_time[term].astype(np.int16)
# user_time = user_time[user_time['user_id']!=839368]

#1.2 creative_id 2481135  4445718
k = 26713
term = 'creative_id'
user_creative_id = pd.read_csv(user_creative_id_test_path)
user_creative_id[term] = user_creative_id[term].apply(lambda x : list(map(lambda y : int(y%k),eval(str(x)))))
# user_creative_id[term] = user_creative_id[term].astype(np.int16)

#1.3 ad_id 2264190 3812200
k = 26713
term = 'ad_id'
user_ad_id = pd.read_csv(user_ad_id_test_path)
user_ad_id[term] = user_ad_id[term].apply(lambda x : list(map(lambda y : int(y%k),eval(str(x)))))
# user_ad_id[term] = user_ad_id[term].astype(np.int16)


#1.4 advertiser_id  52090 62965.0
k = 26713
term = 'advertiser_id'
user_advertiser_id = pd.read_csv(user_advertiser_id_test_path)
user_advertiser_id[term] = user_advertiser_id[term].apply(lambda x : list(map(lambda y : int(y%k),eval(str(x)))))
# user_advertiser_id[term] = user_advertiser_id[term].astype(np.int16)

#1.5 industry 326 335.0 nan
k = 336
term = 'industry'
user_industry = pd.read_csv(user_industry_test_path)
user_industry[term] = user_industry[term].apply(lambda x : list(map(lambda y : int(y%k),eval(rmnan(str(x))))))
# user_industry[term] = user_industry[term].astype(np.int16)

#1.6 product_id 33273 44313.0 nan
k = 26713
term = 'product_id'
user_product_id = pd.read_csv(user_product_id_test_path)
user_product_id[term] = user_product_id[term].apply(lambda x : list(map(lambda y : int(y%k),eval(rmnan(str(x))))))
# user_product_id[term] = user_product_id[term].astype(np.int16)

#1.7 product_category  18 18
k = 19
term = 'product_category'
user_product_category = pd.read_csv(user_product_category_test_path)
user_product_category[term] = user_product_category[term].apply(lambda x : list(map(lambda y : int(y%k),eval(str(x)))))
# user_product_category[term] = user_product_category[term].astype(np.int16)

#1.8 join sequence and to_csv
dfs = [user_time,user_creative_id,user_ad_id,user_advertiser_id,user_industry,user_product_id,user_product_category]
df = dfs[0]
for i in range(1,len(dfs)):
    df = pd.merge(df,dfs[i],on='user_id')

df.to_csv(project+'test/user/user_sequence.csv',index=False)
