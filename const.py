

test_ad_path = 'csvs/test/ad.csv'
test_click_log_path = 'csvs/test/click_log.csv'

train_ad_path = 'csvs/train/ad.csv'
train_click_log_path = 'csvs/train/click_log.csv'
train_user_path = 'csvs/train/user.csv'

raw_joined = 'csvs/raw_joined.csv'
raw_processed = 'csvs/raw_processed.csv'

max_len = 150

features_num_dict = {'total_times':9,'weekend_0':9,'weekend_1':8,'product_category_18':9,'weekday_1':8,'weekday_0':8,'weekday_4':8,'weekday_3':8,'weekday_2':8,'weekday_5':8,'weekday_6':8,'industry_6':9,'product_category_2':9,'product_category_5':9,'industry_319':8,'industry_322':9,'industry_247':9,'industry_54':8,'industry_317':8,'product_category_3':9,'industry_297':7,'industry_238':8,'industry_242':8,'industry_73':8,'product_category_12':7,'industry_88':7,'industry_289':7,'product_category_8':9,'industry_60':8,'industry_248':7,'industry_25':7,'product_category_17':9,'industry_326':8,'industry_246':7,'industry_21':8,'industry_291':7,'industry_5':8,'industry_318':8,'industry_47':8,'industry_296':8,'industry_329':8,'industry_36':8,'industry_40':8,'industry_252':7,'industry_27':7,'industry_26':7,'industry_183':8,'industry_203':6,'industry_202':6,'industry_253':8,'product_category_13':8,'industry_321':9,'industry_288':7,'industry_259':7,'industry_205':6 ,\
    'time_len':max_len,'time_size':92,'creative_id_len':max_len,'creative_id_size':26714,'ad_id_len':max_len,'ad_id_size':26714,'advertiser_id_len':max_len,'advertiser_id_size':26714,'industry_len':max_len,'industry_size':336,'product_id_len':max_len,'product_id_size':26714,'product_category_len':max_len,'product_category_size':19,}






#######

project = '/home/tione/notebook/'




## 1. train file
click_log_path = project +'train/click_log_joined.csv'

user_time_path = project + 'train/user/user_time.csv'
user_creative_id_path = project + 'train/user/user_creative_id.csv'
user_ad_id_path = project + 'train/user/user_ad_id.csv'
user_advertiser_id_path = project + 'train/user/user_advertiser_id.csv'
user_industry_path = project + 'train/user/user_industry.csv'
user_product_id_path = project + 'train/user/user_product_id.csv'
user_product_category_path = project + 'train/user/user_product_category.csv'



## 2. test file

ad_test_path = project + 'test/raw/ad.csv'
click_log_test_path = project + 'test/raw/click_log.csv'
click_log_test_joined = project +'test/click_log_joined.csv'

user_time_test_path = project + 'test/user/user_time.csv'
user_creative_id_test_path = project + 'test/user/user_creative_id.csv'
user_ad_id_test_path = project + 'test/user/user_ad_id.csv'
user_advertiser_id_test_path = project + 'test/user/user_advertiser_id.csv'
user_industry_test_path = project + 'test/user/user_industry.csv'
user_product_id_test_path = project + 'test/user/user_product_id.csv'
user_product_category_test_path = project + 'test/user/user_product_category.csv'

