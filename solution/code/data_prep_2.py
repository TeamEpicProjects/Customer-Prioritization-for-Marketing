# This script merges and consolidates the customer, transaction and product datasets into 1 dataset

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

# Summary and head of dataset ct_3m
print('\n{}\tReading raw data: ct_3m.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = 'ct_3m.csv'
df_ct = pd.read_csv(os.path.join(base_path, in_filename))
df_ct_info = df_info(df_ct)
print('\n{}\t"ct_3m" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_ct.shape[0], df_ct.shape[1], df_ct_info[1]))
print(df_ct_info[0].to_string())
print('\n"ct_3m" dataset head:')
print(df_ct.head().to_string())

# Summary and head of dataset c
print('\n{}\tReading raw data: c.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = 'c.csv'
df_c = pd.read_csv(os.path.join(base_path, in_filename))
df_c_info = df_info(df_c)
print('\n{}\t"c" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_c.shape[0], df_c.shape[1], df_c_info[1]))
print(df_c_info[0].to_string())
print('\n"c" dataset head:')
print(df_c.head().to_string())

# Summary and head of dataset tp
print('\n{}\tReading raw data: tp.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = 'tp.csv'
df_tp = pd.read_csv(os.path.join(base_path, in_filename))
df_tp_info = df_info(df_tp)
print('\n{}\t"tp" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_tp.shape[0], df_tp.shape[1], df_tp_info[1]))
print(df_tp_info[0].to_string())
print('\n"tp" dataset head:')
print(df_tp.head().to_string())

######################

print('\n{}\tProcessing stage 5: merging "c" and "tp" with "ct_3m" ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# Merging the tables c, ct and tp
df_ct_merged = (((df_ct.merge(df_c.drop(columns=['primary_phone', 'secondary_phones']), left_on='cid', right_on='id', how='inner')).drop(columns=['id_y'])).merge(df_tp.drop(columns=['status']), left_on='id_x', right_on='ctid', how='inner')).drop(columns=['id_x'])

# Summary and head of the merged dataset.
df_ct_merged_info = df_info(df_ct_merged)
print('\n{}\t"ct_merged" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_ct_merged.shape[0], df_ct_merged.shape[1], df_ct_merged_info[1]))
print(df_ct_merged_info[0].to_string())
print('\n"ct_merged" dataset head:')
print(df_ct_merged.head().to_string())

######################

print('\n{}\tProcessing stage 6: consolidating "ct_merged" dataset by date and email ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# Consolidating the merged dataset 
df_ct_merged_gb_date_email = df_ct_merged.groupby(['timestamp', 'email'])
df_ct_merged_cb_date_email = pd.DataFrame(df_ct_merged_gb_date_email.groups.keys(), columns=['date', 'email'])

def consolidate_conversion(x):
    """
    Collapses the conversion status to Yes or No.
    """
    xl = x.str.lower()
    if any(xl.str.startswith('purchase')) or any(xl.str.startswith('converted')) or \
            any(xl.str.startswith('y')) or any(xl.str.startswith('processed')) or \
            any(xl.str.startswith('payment_completed')) or any(xl.str.startswith('initiated')) \
            or any(xl.str.startswith('pdf_error')) or any(xl.str.startswith('toprocess')) \
            or any(xl.str.startswith('delivered')):
        return 'Y'
    else:
        return 'N'

print('{}\t\tConsolidating conversion_status ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# According to the consolidate_conversion function
df_ct_merged_cb_date_email['conversion_status'] = df_ct_merged_cb_date_email.parallel_apply(lambda x: consolidate_conversion(df_ct_merged_gb_date_email.get_group((x[0], x[1]))['status']), axis=1)
print('{}\t\tConsolidating profile_submit_count ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# 
df_ct_merged_cb_date_email['profile_submit_count'] = df_ct_merged_cb_date_email.parallel_apply(lambda x: df_ct_merged_gb_date_email.get_group((x[0], x[1])).iloc[0, 5], axis=1)
print('{}\t\tConsolidating transactions_amount ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
df_ct_merged_cb_date_email['transactions_amount'] = df_ct_merged_cb_date_email.parallel_apply(lambda x: df_ct_merged_gb_date_email.get_group((x[0], x[1]))['amount'].values.sum(), axis=1)

out_filename = 'ct_merged_consolidated_3m.csv'
df_ct_merged_cb_date_email.to_csv(os.path.join(base_path, out_filename), index=False)

df_ct_merged_cb_date_email_info = df_info(df_ct_merged_cb_date_email)
print('\n{}\t"ct_merged_cb_date_email" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_ct_merged_cb_date_email.shape[0], df_ct_merged_cb_date_email.shape[1], df_ct_merged_cb_date_email_info[1]))
print(df_ct_merged_cb_date_email_info[0].to_string())
print('\n"ct_merged_cb_date_email" dataset head:')
print(df_ct_merged_cb_date_email.head().to_string())

######################

# /home/ngkpg/anaconda3/envs/pyconda37/bin/python3.7 /home/ngkpg/Documents/Packt_GP/GP1/code/data_prep_2.py
# INFO: Pandarallel will run on 4 workers.
# INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.
#
# 2021-09-19 00:23:36	Reading raw data: ct_3m.csv ...
#
# 2021-09-19 00:23:38	"ct_3m" dataset summary:
# 	3304478 rows x 5 columns | 469.92 MB approx memory usage
#     col_name                 col_type  null_count  nunique
# 0         id    <class 'numpy.int64'>           0  3304478
# 1        cid    <class 'numpy.int64'>           0  1325451
# 2  timestamp            <class 'str'>           0       92
# 3     amount  <class 'numpy.float64'>           0      878
# 4     status            <class 'str'>           0       10
#
# "ct_3m" dataset head:
#      id   cid   timestamp  amount             status
# 0  1196  1356  2021-05-01  7100.0  PAYMENT_COMPLETED
# 1  1197     1  2021-05-01   730.0  PAYMENT_COMPLETED
# 2  1198   160  2021-05-01  1600.0  PAYMENT_COMPLETED
# 3  1199   319  2021-05-01  2247.0  PAYMENT_COMPLETED
# 4  1200   319  2021-05-01    40.0  PAYMENT_COMPLETED
#
# 2021-09-19 00:23:38	Reading raw data: c.csv ...
#
# 2021-09-19 00:23:39	"c" dataset summary:
# 	2295101 rows x 5 columns | 143.52 MB approx memory usage
#                col_name                 col_type  null_count  nunique
# 0                    id    <class 'numpy.int64'>           0  2295101
# 1                 email    <class 'numpy.int64'>           0  2295101
# 2         primary_phone  <class 'numpy.float64'>      793012  1348348
# 3      secondary_phones          <class 'float'>     2180343    97881
# 4  profile_submit_count    <class 'numpy.int64'>           0      413
#
# "c" dataset head:
#    id    email  primary_phone secondary_phones  profile_submit_count
# 0   1   537606           22.0              NaN                   592
# 1   5  1443908            NaN              NaN                     3
# 2   6   534973            NaN           588180                     6
# 3   7  3259797            NaN              NaN                     3
# 4   8  1701404            NaN              NaN                     5
#
# 2021-09-19 00:23:39	Reading raw data: tp.csv ...
#
# 2021-09-19 00:23:42	"tp" dataset summary:
# 	4179024 rows x 4 columns | 647.58 MB approx memory usage
#    col_name               col_type  null_count  nunique
# 0      ctid  <class 'numpy.int64'>           0  4170263
# 1   variant          <class 'str'>           0        6
# 2  language          <class 'str'>        1824       30
# 3    status        <class 'float'>     4127969        5
#
# "tp" dataset head:
#    ctid  variant language status
# 0     4  premium      tel    NaN
# 1     5  premium      eng    NaN
# 2     6  premium      eng    NaN
# 3     7  premium      eng    NaN
# 4     8  premium      eng    NaN
#
# 2021-09-19 00:23:42	Processing stage 5: merging "c" and "tp" with "ct_3m" ...
#
# 2021-09-19 00:23:46	"ct_merged" dataset summary:
# 	3308922 rows x 9 columns | 1133.47 MB approx memory usage
#                col_name                 col_type  null_count  nunique
# 0                   cid    <class 'numpy.int64'>           0  1325200
# 1             timestamp            <class 'str'>           0       92
# 2                amount  <class 'numpy.float64'>           0      877
# 3                status            <class 'str'>           0       10
# 4                 email    <class 'numpy.int64'>           0  1325200
# 5  profile_submit_count    <class 'numpy.int64'>           0      413
# 6                  ctid    <class 'numpy.int64'>           0  3301429
# 7               variant            <class 'str'>           0        6
# 8              language            <class 'str'>        1411       29
#
# "ct_merged" dataset head:
#     cid   timestamp  amount             status  email  profile_submit_count     ctid  variant language
# 0  1356  2021-05-01  7100.0  PAYMENT_COMPLETED  94625                    10     1196  premium      eng
# 1  1356  2021-05-19     0.0                  N  94625                    10   599717    basic      eng
# 2  1356  2021-07-27     0.0                  N  94625                    10  3153363    basic      eng
# 3  1356  2021-07-30     0.0                  N  94625                    10  3254148    basic      eng
# 4  1356  2021-07-30   999.0          PROCESSED  94625                    10  3254480  premium      eng
#
# 2021-09-19 00:23:46	Processing stage 6: consolidating "ct_merged" dataset by date and email ...
# 2021-09-19 00:24:05		Consolidating conversion_status ...
# 2021-09-19 00:34:18		Consolidating profile_submit_count ...
# 2021-09-19 00:36:05		Consolidating transactions_amount ...
#
# 2021-09-19 00:37:56	"ct_merged_cb_date_email" dataset summary:
# 	1812271 rows x 5 columns | 257.52 MB approx memory usage
#                col_name                 col_type  null_count  nunique
# 0                  date            <class 'str'>           0       92
# 1                 email    <class 'numpy.int64'>           0  1325200
# 2     conversion_status            <class 'str'>           0        2
# 3  profile_submit_count    <class 'numpy.int64'>           0      413
# 4   transactions_amount  <class 'numpy.float64'>           0     2320
#
# "ct_merged_cb_date_email" dataset head:
#          date  email conversion_status  profile_submit_count  transactions_amount
# 0  2021-05-01    125                 N                    51                  0.0
# 1  2021-05-01    141                 N                  4347                  0.0
# 2  2021-05-01    195                 N                  1048                  0.0
# 3  2021-05-01    269                 N                    10                  0.0
# 4  2021-05-01    513                 N                    14                  0.0

######################
