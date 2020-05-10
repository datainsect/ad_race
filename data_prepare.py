import pandas  as pd

project = '/home/tione/notebook/'

## 1.process train file
user_time_path = project + 'train/user/user_time.csv'
user_creative_id_path = project + 'train/user/user_creative_id.csv'
user_ad_id_path = project + 'train/user/user_ad_id.csv'
user_industry_path = project + 'train/user/user_industry.csv'
user_advertiser_id_path = project + 'train/user/user_advertiser_id.csv'
user_product_id_path = project + 'train/user/user_product_id.csv'
user_product_category_path = project + 'train/user/user_product_category.csv'

user_time_path = pd.read_csv('user_time_path')


