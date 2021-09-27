from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
import pandas as pd
import datetime
import os


pandarallel.initialize(progress_bar=False, nb_workers=4)

base_path = os.path.dirname(os.path.realpath(__file__))

def df_info(df):
    col_name_list = list(df.columns)
    col_type_list = [type(col) for col in df.iloc[0, :]]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_memory_usage_list = [df[col].memory_usage(deep=True) for col in col_name_list]
    df_total_memory_usage = sum(col_memory_usage_list) / 1048576
    return pd.DataFrame({'col_name': col_name_list, 'col_type': col_type_list, 'null_count': col_null_count_list, 'nunique': col_unique_count_list}), df_total_memory_usage

######################

print('\n{}\tReading dataset: bs_ct_merged_consolidated_3m.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = '../data/sanitized/processed_base/bs_ct_merged_consolidated_3m.csv'
df_base_data = pd.read_csv(os.path.join(base_path, in_filename))
df_base_data.date = df_base_data.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
df_base_data_info = df_info(df_base_data)
print('\n{}\t"base_data" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_base_data.shape[0], df_base_data.shape[1], df_base_data_info[1]))
print(df_base_data_info[0].to_string())
print('\n"base_data" dataset head:')
print(df_base_data.head().to_string())

######################

df_base_data_dev, df_base_data_ops = train_test_split(df_base_data, random_state=0, stratify=df_base_data.conversion_status)

out_filename = 'base_data_dev_3m.csv'
df_base_data_dev.to_csv(os.path.join(base_path, out_filename), index=False)

out_filename = 'base_data_ops_3m.csv'
df_base_data_ops.to_csv(os.path.join(base_path, out_filename), index=False)

df_base_data_dev_info = df_info(df_base_data_dev)
print('\n{}\t"base_data_dev" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_base_data_dev.shape[0], df_base_data_dev.shape[1], df_base_data_dev_info[1]))
print(df_base_data_dev_info[0].to_string())
print('\n"base_data_dev" dataset head:')
print(df_base_data_dev.head().to_string())
label_value_counts = df_base_data_dev.conversion_status.value_counts()
print('\tClass Distribution for base_data_dev.conversion_status: 0 = {}, 1 = {}'.format(label_value_counts.loc[0], label_value_counts.loc[1]))

df_base_data_ops_info = df_info(df_base_data_ops)
print('\n{}\t"base_data_ops" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_base_data_ops.shape[0], df_base_data_ops.shape[1], df_base_data_ops_info[1]))
print(df_base_data_ops_info[0].to_string())
print('\n"base_data_ops" dataset head:')
print(df_base_data_ops.head().to_string())
label_value_counts = df_base_data_ops.conversion_status.value_counts()
print('\tClass Distribution for base_data_ops.conversion_status: 0 = {}, 1 = {}'.format(label_value_counts.loc[0], label_value_counts.loc[1]))

######################

# /home/ngkpg/anaconda3/envs/pyconda37/bin/python3.7 /home/ngkpg/Documents/Packt_GP/GP1/code/split_base_data.py
# INFO: Pandarallel will run on 4 workers.
# INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.
#
# 2021-09-19 05:15:52	Reading dataset: bs_ct_merged_consolidated_3m.csv ...
#
# 2021-09-19 05:15:58	"base_data" dataset summary:
# 	1064216 rows x 16 columns | 129.91 MB approx memory usage
#                 col_name                                            col_type  null_count  nunique
# 0                   date  <class 'pandas._libs.tslibs.timestamps.Timestamp'>           0       92
# 1                  email                               <class 'numpy.int64'>           0   824412
# 2         count_sessions                               <class 'numpy.int64'>           0       56
# 3       sum_beacon_value                               <class 'numpy.int64'>           0     2549
# 4    nunique_beacon_type                               <class 'numpy.int64'>           0       62
# 5        count_user_stay                               <class 'numpy.int64'>           0      237
# 6      count_pay_attempt                               <class 'numpy.int64'>           0       45
# 7        count_buy_click                               <class 'numpy.int64'>           0       42
# 8         nunique_gender                               <class 'numpy.int64'>           0        3
# 9            nunique_dob                               <class 'numpy.int64'>           0       42
# 10      nunique_language                               <class 'numpy.int64'>           0        8
# 11   nunique_report_type                               <class 'numpy.int64'>           0       13
# 12        nunique_device                               <class 'numpy.int64'>           0        5
# 13     conversion_status                               <class 'numpy.int64'>           0        2
# 14  profile_submit_count                               <class 'numpy.int64'>           0      360
# 15   transactions_amount                             <class 'numpy.float64'>           0     1410
#
# "base_data" dataset head:
#         date  email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device  conversion_status  profile_submit_count  transactions_amount
# 0 2021-05-01    125               3                30                    3               12                  0                0               2            2                 1                    1               1                  0                    51                  0.0
# 1 2021-05-01    141               5                39                    5               16                  0                0               2            5                 4                    2               2                  0                  4347              14962.0
# 2 2021-05-01    195               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                  1048               1028.0
# 3 2021-05-01    645               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     5                  0.0
# 4 2021-05-01    798               1                 3                    1                2                  0                0               1            1                 1                    1               1                  0                    88                  0.0
#
# 2021-09-19 05:16:05	"base_data_dev" dataset summary:
# 	798162 rows x 16 columns | 194.86 MB approx memory usage
#                 col_name                                            col_type  null_count  nunique
# 0                   date  <class 'pandas._libs.tslibs.timestamps.Timestamp'>           0       92
# 1                  email                               <class 'numpy.int64'>           0   643967
# 2         count_sessions                               <class 'numpy.int64'>           0       52
# 3       sum_beacon_value                               <class 'numpy.int64'>           0     2257
# 4    nunique_beacon_type                               <class 'numpy.int64'>           0       58
# 5        count_user_stay                               <class 'numpy.int64'>           0      228
# 6      count_pay_attempt                               <class 'numpy.int64'>           0       44
# 7        count_buy_click                               <class 'numpy.int64'>           0       40
# 8         nunique_gender                               <class 'numpy.int64'>           0        3
# 9            nunique_dob                               <class 'numpy.int64'>           0       35
# 10      nunique_language                               <class 'numpy.int64'>           0        8
# 11   nunique_report_type                               <class 'numpy.int64'>           0       12
# 12        nunique_device                               <class 'numpy.int64'>           0        4
# 13     conversion_status                               <class 'numpy.int64'>           0        2
# 14  profile_submit_count                               <class 'numpy.int64'>           0      360
# 15   transactions_amount                             <class 'numpy.float64'>           0     1288
#
# "base_data_dev" dataset head:
#               date    email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device  conversion_status  profile_submit_count  transactions_amount
# 1031892 2021-07-29  3237641               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     2                  0.0
# 97957   2021-05-15  2637791               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     1                  0.0
# 350802  2021-06-11  2807521               1               106                    2               14                  0                1               1            1                 1                    1               1                  0                     2                  0.0
# 226363  2021-06-02   832889               1                14                    1                5                  0                0               1            1                 1                    1               1                  0                     8                  0.0
# 661228  2021-07-05  2624659               1                 2                    2                1                  0                0               1            1                 1                    1               1                  0                     8                  0.0
# 	Class Distribution for base_data_dev.conversion_status: 0 = 785538, 1 = 12624
#
# 2021-09-19 05:16:05	"base_data_ops" dataset summary:
# 	266054 rows x 16 columns | 64.95 MB approx memory usage
#                 col_name                                            col_type  null_count  nunique
# 0                   date  <class 'pandas._libs.tslibs.timestamps.Timestamp'>           0       92
# 1                  email                               <class 'numpy.int64'>           0   239453
# 2         count_sessions                               <class 'numpy.int64'>           0       46
# 3       sum_beacon_value                               <class 'numpy.int64'>           0     1387
# 4    nunique_beacon_type                               <class 'numpy.int64'>           0       46
# 5        count_user_stay                               <class 'numpy.int64'>           0      171
# 6      count_pay_attempt                               <class 'numpy.int64'>           0       24
# 7        count_buy_click                               <class 'numpy.int64'>           0       20
# 8         nunique_gender                               <class 'numpy.int64'>           0        3
# 9            nunique_dob                               <class 'numpy.int64'>           0       31
# 10      nunique_language                               <class 'numpy.int64'>           0        8
# 11   nunique_report_type                               <class 'numpy.int64'>           0        8
# 12        nunique_device                               <class 'numpy.int64'>           0        4
# 13     conversion_status                               <class 'numpy.int64'>           0        2
# 14  profile_submit_count                               <class 'numpy.int64'>           0      349
# 15   transactions_amount                             <class 'numpy.float64'>           0      817
#
# "base_data_ops" dataset head:
#              date    email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device  conversion_status  profile_submit_count  transactions_amount
# 626887 2021-07-02  2982181               1                12                    2                4                  0                0               1            1                 1                    1               1                  0                     1                  0.0
# 999796 2021-07-27  3216150               2                16                    2                7                  0                0               1            1                 1                    1               1                  0                     2                  0.0
# 693378 2021-07-07  2977312               1                36                    1                8                  0                0               1            1                 1                    1               1                  0                     3                  0.0
# 341951 2021-06-11  2594563               5                21                    5               11                  0                0               1            2                 1                    1               1                  0                    15                  0.0
# 733366 2021-07-09  3050701               3                23                    5                9                  0                1               1            1                 1                    1               1                  0                     3                  0.0
# 	Class Distribution for base_data_ops.conversion_status: 0 = 261846, 1 = 4208
