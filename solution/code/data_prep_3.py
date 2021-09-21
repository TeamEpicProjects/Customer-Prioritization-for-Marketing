# This script merges and consolidates the 2 datasets from data_prep_1 and dat_prep_2 into 1 dataset

from pandarallel import pandarallel
import pandas as pd
import datetime
import os


pandarallel.initialize(progress_bar=False, nb_workers=4)

base_path = os.path.dirname(os.path.realpath(__file__))

def df_info(df):
    """
    Function to get information about a dataframe
    """
    col_name_list = list(df.columns)
    col_type_list = [type(col) for col in df.iloc[0, :]]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_memory_usage_list = [df[col].memory_usage(deep=True) for col in col_name_list]
    df_total_memory_usage = sum(col_memory_usage_list) / 1048576
    return pd.DataFrame({'col_name': col_name_list, 'col_type': col_type_list, 'null_count': col_null_count_list, 'nunique': col_unique_count_list}), df_total_memory_usage

######################

print('\n{}\tReading dataset: bs_merged_consolidated_3m.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = '../data/sanitized/processed_base/bs_merged_consolidated_3m.csv'
df_bs_merged_consolidated = pd.read_csv(os.path.join(base_path, in_filename))
df_bs_merged_consolidated_info = df_info(df_bs_merged_consolidated)
print('\n{}\t"bs_merged_consolidated" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_bs_merged_consolidated.shape[0], df_bs_merged_consolidated.shape[1], df_bs_merged_consolidated_info[1]))
print(df_bs_merged_consolidated_info[0].to_string())
print('\n"bs_merged_consolidated" dataset head:')
print(df_bs_merged_consolidated.head().to_string())

print('\n{}\tReading dataset: ct_merged_consolidated_3m.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = '../data/sanitized/processed_base/ct_merged_consolidated_3m.csv'
df_ct_merged_consolidated = pd.read_csv(os.path.join(base_path, in_filename))
df_ct_merged_consolidated_info = df_info(df_ct_merged_consolidated)
print('\n{}\t"ct_merged_consolidated" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_ct_merged_consolidated.shape[0], df_ct_merged_consolidated.shape[1], df_ct_merged_consolidated_info[1]))
print(df_ct_merged_consolidated_info[0].to_string())
print('\n"ct_merged_consolidated" dataset head:')
print(df_ct_merged_consolidated.head().to_string())
label_value_counts = df_ct_merged_consolidated.conversion_status.value_counts()
print('\n"ct_merged_consolidated.conversion_status" N = {}, Y = {}'.format(label_value_counts.loc['N'], label_value_counts.loc['Y']))
print('"ct_merged_consolidated.profile_submit_count" min = {}, max = {}'.format(df_ct_merged_consolidated.profile_submit_count.min(), df_ct_merged_consolidated.profile_submit_count.max()))

######################

print('\n{}\tProcessing stage 7: merging "bs_merged_consolidated" and "ct_merged_consolidated" ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

df_ct_merged_consolidated_gb_email = df_ct_merged_consolidated.groupby(['email'])
df_bs_ct_merged_consolidated = df_bs_merged_consolidated.copy()

def merge_conversion(df_y, date_x):
    """
    Function to collapse the conversion status of each customer during a period of 3 days
       starting the day the customer visited our website and submitted details for a free horoscope report
    """
    date_x = datetime.datetime.strptime(date_x, '%Y-%m-%d')
    df_y.date = df_y.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    date_x_plus_2 = date_x + datetime.timedelta(days=2)
    if any((df_y.loc[(df_y.date >= date_x) & (df_y.date <= date_x_plus_2)]).conversion_status.str.startswith('Y')):
        return 1
    else:
        return 0

print('{}\t\tConsolidating conversion_status ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# Consolidated keeping in mind the time window of 3 days
df_bs_ct_merged_consolidated['conversion_status'] = df_bs_ct_merged_consolidated.parallel_apply(lambda x: merge_conversion(df_ct_merged_consolidated_gb_email.get_group(x['email'])[['date', 'conversion_status']], x['date']) if x['email'] in df_ct_merged_consolidated_gb_email.groups.keys() else 0, axis=1)
print('{}\t\tConsolidating profile_submit_count ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# Profile submit count is the same value for all the customers with same id (it's in the ct table)
df_bs_ct_merged_consolidated['profile_submit_count'] = df_bs_ct_merged_consolidated.parallel_apply(lambda x: df_ct_merged_consolidated_gb_email.get_group(x['email']).iloc[0, 3] if x['email'] in df_ct_merged_consolidated_gb_email.groups.keys() else 0, axis=1)
print('{}\t\tConsolidating transactions_amount ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# Include the amount value if the record is present in the beacon-session table
df_bs_ct_merged_consolidated['transactions_amount'] = df_bs_ct_merged_consolidated.parallel_apply(lambda x: df_ct_merged_consolidated_gb_email.get_group(x['email'])['transactions_amount'].values.sum() if x['email'] in df_ct_merged_consolidated_gb_email.groups.keys() else -1.0, axis=1)

out_filename = 'bs_ct_merged_consolidated_3m.csv'
df_bs_ct_merged_consolidated.to_csv(os.path.join(base_path, out_filename), index=False)

df_bs_ct_merged_consolidated_info = df_info(df_bs_ct_merged_consolidated)
print('\n{}\t"bs_ct_merged_consolidated" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_bs_ct_merged_consolidated.shape[0], df_bs_ct_merged_consolidated.shape[1], df_bs_ct_merged_consolidated_info[1]))
print(df_bs_ct_merged_consolidated_info[0].to_string())
print('\n"bs_ct_merged_consolidated" dataset head:')
print(df_bs_ct_merged_consolidated.head().to_string())
label_value_counts = df_bs_ct_merged_consolidated.conversion_status.value_counts()
print('\tClass Distribution for bs_ct_merged_consolidated.conversion_status: 0 = {}, 1 = {}'.format(label_value_counts.loc[0], label_value_counts.loc[1]))

######################

# /home/ngkpg/anaconda3/envs/pyconda37/bin/python3.7 /home/ngkpg/Documents/Packt_GP/GP1/code/data_prep_3.py
# INFO: Pandarallel will run on 4 workers.
# INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.
#
# 2021-09-19 03:18:52	Reading dataset: bs_merged_consolidated_3m.csv ...
#
# 2021-09-19 03:18:53	"bs_merged_consolidated" dataset summary:
# 	1064216 rows x 13 columns | 165.43 MB approx memory usage
#                col_name               col_type  null_count  nunique
# 0                  date          <class 'str'>           0       92
# 1                 email  <class 'numpy.int64'>           0   824412
# 2        count_sessions  <class 'numpy.int64'>           0       56
# 3      sum_beacon_value  <class 'numpy.int64'>           0     2549
# 4   nunique_beacon_type  <class 'numpy.int64'>           0       62
# 5       count_user_stay  <class 'numpy.int64'>           0      237
# 6     count_pay_attempt  <class 'numpy.int64'>           0       45
# 7       count_buy_click  <class 'numpy.int64'>           0       42
# 8        nunique_gender  <class 'numpy.int64'>           0        3
# 9           nunique_dob  <class 'numpy.int64'>           0       42
# 10     nunique_language  <class 'numpy.int64'>           0        8
# 11  nunique_report_type  <class 'numpy.int64'>           0       13
# 12       nunique_device  <class 'numpy.int64'>           0        5
#
# "bs_merged_consolidated" dataset head:
#          date  email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device
# 0  2021-05-01    125               3                30                    3               12                  0                0               2            2                 1                    1               1
# 1  2021-05-01    141               5                39                    5               16                  0                0               2            5                 4                    2               2
# 2  2021-05-01    195               1                10                    1                4                  0                0               1            1                 1                    1               1
# 3  2021-05-01    645               1                10                    1                4                  0                0               1            1                 1                    1               1
# 4  2021-05-01    798               1                 3                    1                2                  0                0               1            1                 1                    1               1
#
# 2021-09-19 03:18:53	Reading dataset: ct_merged_consolidated_3m.csv ...
#
# 2021-09-19 03:18:54	"ct_merged_consolidated" dataset summary:
# 	1812271 rows x 5 columns | 257.52 MB approx memory usage
#                col_name                 col_type  null_count  nunique
# 0                  date            <class 'str'>           0       92
# 1                 email    <class 'numpy.int64'>           0  1325200
# 2     conversion_status            <class 'str'>           0        2
# 3  profile_submit_count    <class 'numpy.int64'>           0      413
# 4   transactions_amount  <class 'numpy.float64'>           0     2320
#
# "ct_merged_consolidated" dataset head:
#          date  email conversion_status  profile_submit_count  transactions_amount
# 0  2021-05-01    125                 N                    51                  0.0
# 1  2021-05-01    141                 N                  4347                  0.0
# 2  2021-05-01    195                 N                  1048                  0.0
# 3  2021-05-01    269                 N                    10                  0.0
# 4  2021-05-01    513                 N                    14                  0.0
#
# "ct_merged_consolidated.conversion_status" N = 1780161, Y = 32110
# "ct_merged_consolidated.profile_submit_count" min = 1, max = 9842
#
# 2021-09-19 03:18:54	Processing stage 7: merging "bs_merged_consolidated" and "ct_merged_consolidated" ...
# 2021-09-19 03:18:54		Consolidating conversion_status ...
# 2021-09-19 03:27:31		Consolidating profile_submit_count ...
# 2021-09-19 03:28:45		Consolidating transactions_amount ...
#
# 2021-09-19 03:30:00	"bs_ct_merged_consolidated" dataset summary:
# 	1064216 rows x 16 columns | 189.79 MB approx memory usage
#                 col_name                 col_type  null_count  nunique
# 0                   date            <class 'str'>           0       92
# 1                  email    <class 'numpy.int64'>           0   824412
# 2         count_sessions    <class 'numpy.int64'>           0       56
# 3       sum_beacon_value    <class 'numpy.int64'>           0     2549
# 4    nunique_beacon_type    <class 'numpy.int64'>           0       62
# 5        count_user_stay    <class 'numpy.int64'>           0      237
# 6      count_pay_attempt    <class 'numpy.int64'>           0       45
# 7        count_buy_click    <class 'numpy.int64'>           0       42
# 8         nunique_gender    <class 'numpy.int64'>           0        3
# 9            nunique_dob    <class 'numpy.int64'>           0       42
# 10      nunique_language    <class 'numpy.int64'>           0        8
# 11   nunique_report_type    <class 'numpy.int64'>           0       13
# 12        nunique_device    <class 'numpy.int64'>           0        5
# 13     conversion_status    <class 'numpy.int64'>           0        2
# 14  profile_submit_count    <class 'numpy.int64'>           0      360
# 15   transactions_amount  <class 'numpy.float64'>           0     1410
#
# "bs_ct_merged_consolidated" dataset head:
#          date  email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device  conversion_status  profile_submit_count  transactions_amount
# 0  2021-05-01    125               3                30                    3               12                  0                0               2            2                 1                    1               1                  0                    51                  0.0
# 1  2021-05-01    141               5                39                    5               16                  0                0               2            5                 4                    2               2                  0                  4347              14962.0
# 2  2021-05-01    195               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                  1048               1028.0
# 3  2021-05-01    645               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     5                  0.0
# 4  2021-05-01    798               1                 3                    1                2                  0                0               1            1                 1                    1               1                  0                    88                  0.0
# 	Class Distribution for bs_ct_merged_consolidated.conversion_status: 0 = 1047384, 1 = 16832
