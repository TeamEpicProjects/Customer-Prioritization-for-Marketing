# This script merges and consolidates the beacons and sessions datasets into 1 dataset

from pandarallel import pandarallel
import pandas as pd
import datetime
import sys
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

# Due to memory constraint, we split this script using an argument
#   We are simply going to check if an argument was supplied or not
#   If not supplied, we shall execute processing stages 1, 2 & 3
#   If supplied, we shall execute only stage 4

if len(sys.argv) < 2:
    # Reading the dataset b_3m and printing its basic summary
    print('\n{}\tReading raw data: b_3m.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    in_filename = '../data/sanitized/subset/b_3m.csv'
    df_b = pd.read_csv(os.path.join(base_path, in_filename))
    df_b_info = df_info(df_b)
    print('\n{}\t"b_3m" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_b.shape[0], df_b.shape[1], df_b_info[1]))
    print(df_b_info[0].to_string())
    print('\n"b_3m" dataset head:')
    print(df_b.head().to_string())
    
    # Reading the dataset s and printing its basic summary
    print('\n{}\tReading raw data: s.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    in_filename = '../data/sanitized/s.csv'
    df_s = pd.read_csv(os.path.join(base_path, in_filename))
    df_s_info = df_info(df_s)
    print('\n{}\t"s" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_s.shape[0], df_s.shape[1], df_s_info[1]))
    print(df_s_info[0].to_string())
    print('\n"s" dataset head:')
    print(df_s.head().to_string())

    ######################

    # Dropping null values and converting column types for dataset b
    print('\n{}\tProcessing stage 1: "b_3m" dataset: dropping rows with na, converting column types ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_b.dropna(inplace=True)
    df_b.uuid = df_b.uuid.parallel_apply(lambda x: str(int(x)))
    df_b.beacon_value = df_b.beacon_value.parallel_apply(lambda x: int(x))
    df_b_info = df_info(df_b)
    print('\n{}\t"b_3m" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_b.shape[0], df_b.shape[1], df_b_info[1]))
    print(df_b_info[0].to_string())
    print('\n"b_3m" dataset head:')
    print(df_b.head().to_string())
    
    # Dropping null values and converting column types for dataset s
    print('\n{}\tProcessing stage 1: "s" dataset: dropping rows with na, dropping columns, converting column types ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_s.dropna(inplace=True)
    df_s.drop(columns=['status'], inplace=True)
    df_s.uuid = df_s.uuid.parallel_apply(lambda x: str(int(x)))
    df_s.phone = df_s.phone.parallel_apply(lambda x: str(int(x)))
    df_s.email = df_s.email.parallel_apply(lambda x: str(int(x)))
    df_s.log_date = df_s.log_date.parallel_apply(lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d').date())
    df_s_info = df_info(df_s)
    print('\n{}\t"s" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_s.shape[0], df_s.shape[1], df_s_info[1]))
    print(df_s_info[0].to_string())
    print('\n"s" dataset head:')
    print(df_s.head().to_string())

    ######################

    print('\n{}\tProcessing stage 2: consolidating "b_3m" dataset by date and uuid ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    # Preparing to consolidate the dataset b_3m
    df_b_gb_date_uuid = df_b.groupby(['log_date', 'uuid'])
    df_b_cb_date_uuid = pd.DataFrame(df_b_gb_date_uuid.groups.keys(), columns=['date', 'uuid'])

    # Feature consolidation
    print('{}\t\tConsolidating sum_beacon_value ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_b_cb_date_uuid['sum_beacon_value'] = df_b_cb_date_uuid.parallel_apply(lambda x: df_b_gb_date_uuid.get_group((x[0], x[1]))['beacon_value'].values.sum(), axis=1)
    print('{}\t\tConsolidating nunique_beacon_type ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_b_cb_date_uuid['nunique_beacon_type'] = df_b_cb_date_uuid.parallel_apply(lambda x: df_b_gb_date_uuid.get_group((x[0], x[1]))['beacon_type'].nunique(), axis=1)
    print('{}\t\tConsolidating count_user_stay ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_b_cb_date_uuid['count_user_stay'] = df_b_cb_date_uuid.parallel_apply(lambda x: (df_b_gb_date_uuid.get_group((x[0], x[1]))['beacon_type'].values == 'user_stay').sum(), axis=1)
    print('{}\t\tConsolidating count_pay_attempt ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_b_cb_date_uuid['count_pay_attempt'] = df_b_cb_date_uuid.parallel_apply(lambda x: df_b_gb_date_uuid.get_group((x[0], x[1]))['beacon_type'].str.contains('pay').sum(), axis=1)
    print('{}\t\tConsolidating count_buy_click ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_b_cb_date_uuid['count_buy_click'] = df_b_cb_date_uuid.parallel_apply(lambda x: df_b_gb_date_uuid.get_group((x[0], x[1]))['beacon_type'].str.contains('buy|bottom').sum(), axis=1)
    
    # Printing the summary of the consolidated beacons table
    df_b_cb_date_uuid_info = df_info(df_b_cb_date_uuid)
    print('\n{}\t"b_3m_cb_date_uuid" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_b_cb_date_uuid.shape[0], df_b_cb_date_uuid.shape[1], df_b_cb_date_uuid_info[1]))
    print(df_b_cb_date_uuid_info[0].to_string())
    print('\n"b_3m_cb_date_uuid" dataset head:')
    print(df_b_cb_date_uuid.head().to_string())

    ######################

    # Merging the consolidated dataset b with s
    print('\n{}\tProcessing stage 3: merging "s" with "b_3m_cb_date_uuid" ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    df_bs_merged = df_b_cb_date_uuid.merge(df_s.drop(columns=['log_date']), on='uuid', how='inner')

    out_filename = '../data/sanitized/processed_base/bs_merged_3m.csv'
    df_bs_merged.to_csv(os.path.join(base_path, out_filename), index=False)

    df_bs_merged_info = df_info(df_bs_merged)
    print('\n{}\t"bs_merged" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_bs_merged.shape[0], df_bs_merged.shape[1], df_bs_merged_info[1]))
    print(df_bs_merged_info[0].to_string())
    print('\n"bs_merged" dataset head:')
    print(df_bs_merged.head().to_string())

else:
    
    # Reading the consolidated dataset and printing its summary
    print('\n{}\tReading dataset: bs_merged_3m.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    in_filename = '../data/sanitized/processed_base/bs_merged_3m.csv'
    df_bs_merged = pd.read_csv(os.path.join(base_path, in_filename))
    df_bs_merged_info = df_info(df_bs_merged)
    print('\n{}\t"bs_merged" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_bs_merged.shape[0], df_bs_merged.shape[1], df_bs_merged_info[1]))
    print(df_bs_merged_info[0].to_string())
    print('\n"bs_merged" dataset head:')
    print(df_bs_merged.head().to_string())

    ######################
    
    # Consolidating the merged dataset by date and email
    print('\n{}\tProcessing stage 4: consolidating "bs_merged" dataset by date and email ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    df_bs_merged_gb_date_email = df_bs_merged.groupby(['date', 'email'])
    df_bs_merged_cb_date_email = pd.DataFrame(df_bs_merged_gb_date_email.groups.keys(), columns=['date', 'email'])

    print('{}\t\tConsolidating count_sessions ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['count_sessions'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: len(df_bs_merged_gb_date_email.get_group((x[0], x[1]))), axis=1)
    print('{}\t\tConsolidating sum_beacon_value ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['sum_beacon_value'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['sum_beacon_value'].values.sum(), axis=1)
    print('{}\t\tConsolidating nunique_beacon_type ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['nunique_beacon_type'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['nunique_beacon_type'].values.sum(), axis=1)
    print('{}\t\tConsolidating count_user_stay ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['count_user_stay'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['count_user_stay'].values.sum(), axis=1)
    print('{}\t\tConsolidating count_pay_attempt ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['count_pay_attempt'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['count_pay_attempt'].values.sum(), axis=1)
    print('{}\t\tConsolidating count_buy_click ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['count_buy_click'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['count_buy_click'].values.sum(), axis=1)
    print('{}\t\tConsolidating nunique_gender ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['nunique_gender'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['gender'].nunique(), axis=1)
    print('{}\t\tConsolidating nunique_dob ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['nunique_dob'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['dob'].nunique(), axis=1)
    print('{}\t\tConsolidating nunique_language ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['nunique_language'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['language'].nunique(), axis=1)
    print('{}\t\tConsolidating nunique_report_type ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['nunique_report_type'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['report_type'].nunique(), axis=1)
    print('{}\t\tConsolidating nunique_device ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df_bs_merged_cb_date_email['nunique_device'] = df_bs_merged_cb_date_email.parallel_apply(lambda x: df_bs_merged_gb_date_email.get_group((x[0], x[1]))['device'].nunique(), axis=1)

    out_filename = '../data/sanitized/processed_base/bs_merged_consolidated_3m.csv'
    df_bs_merged_cb_date_email.to_csv(os.path.join(base_path, out_filename), index=False)

    df_bs_merged_cb_date_email_info = df_info(df_bs_merged_cb_date_email)
    print('\n{}\t"bs_merged_cb_date_email" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_bs_merged_cb_date_email.shape[0], df_bs_merged_cb_date_email.shape[1], df_bs_merged_cb_date_email_info[1]))
    print(df_bs_merged_cb_date_email_info[0].to_string())
    print('\n"bs_merged_cb_date_email" dataset head:')
    print(df_bs_merged_cb_date_email.head().to_string())


######################

# /home/ngkpg/anaconda3/envs/pyconda37/bin/python3.7 /home/ngkpg/Documents/Packt_GP/GP1/code/data_prep_1.py
# INFO: Pandarallel will run on 4 workers.
# INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.
#
# 2021-09-18 22:05:25	Reading raw data: b_3m.csv ...
#
# 2021-09-18 22:05:29	"b_3m" dataset summary:
# 	6970265 rows x 4 columns | 993.62 MB approx memory usage
#        col_name                 col_type  null_count  nunique
# 0          uuid  <class 'numpy.float64'>           2  1794005
# 1   beacon_type            <class 'str'>           2       42
# 2  beacon_value  <class 'numpy.float64'>           2      277
# 3      log_date            <class 'str'>           0       92
#
# "b_3m" dataset head:
#         uuid     beacon_type  beacon_value    log_date
# 0  8264419.0       user_stay           2.0  2021-05-01
# 1  8264429.0  masked_content           1.0  2021-05-01
# 2  8264423.0       user_stay           2.0  2021-05-01
# 3  8264430.0   bottom_banner           1.0  2021-05-01
# 4  8264421.0       user_stay           3.0  2021-05-01
#
# 2021-09-18 22:05:29	Reading raw data: s.csv ...
#
# 2021-09-18 22:05:50	"s" dataset summary:
# 	9095602 rows x 10 columns | 3655.10 MB approx memory usage
#       col_name                 col_type  null_count  nunique
# 0         uuid    <class 'numpy.int64'>           0  9095602
# 1        phone  <class 'numpy.float64'>         977  3399997
# 2       status    <class 'numpy.int64'>           0        1
# 3       gender            <class 'str'>        4765        6
# 4          dob            <class 'str'>          20    36934
# 5     language            <class 'str'>         398       17
# 6        email  <class 'numpy.float64'>         733  3259793
# 7  report_type            <class 'str'>          70       81
# 8       device            <class 'str'>         187        5
# 9     log_date            <class 'str'>           0  8285461
#
# "s" dataset head:
#        uuid     phone  status  gender       dob language  email report_type  device             log_date
# 0  10058150     145.0       1    Male  00000000      TAM    0.0       LS-MT  mobile  2019-02-26 16:07:25
# 1         0     145.0       1    Male  00000000      TAM    0.0       LS-MT  mobile  2019-02-26 16:12:08
# 2         1     145.0       1    Male  00000000      TAM    0.0       LS-MT  mobile  2019-02-26 16:33:00
# 3  10058153  607734.0       1  Female  00000000      TEL    1.0       LS-MP  mobile  2019-02-26 16:44:19
# 4        26  607735.0       1  Female  00000000      TAM    2.0       LS-MT  mobile  2019-02-26 16:44:32
#
# 2021-09-18 22:05:50	Processing stage 1: "b_3m" dataset: dropping rows with na, converting column types ...
#
# 2021-09-18 22:05:58	"b_3m" dataset summary:
# 	6970263 rows x 4 columns | 1578.80 MB approx memory usage
#        col_name               col_type  null_count  nunique
# 0          uuid          <class 'str'>           0  1794005
# 1   beacon_type          <class 'str'>           0       42
# 2  beacon_value  <class 'numpy.int64'>           0      277
# 3      log_date          <class 'str'>           0       92
#
# "b_3m" dataset head:
#       uuid     beacon_type  beacon_value    log_date
# 0  8264419       user_stay             2  2021-05-01
# 1  8264429  masked_content             1  2021-05-01
# 2  8264423       user_stay             2  2021-05-01
# 3  8264430   bottom_banner             1  2021-05-01
# 4  8264421       user_stay             3  2021-05-01
#
# 2021-09-18 22:05:58	Processing stage 1: "s" dataset: dropping rows with na, dropping columns, converting column types ...
#
# 2021-09-18 22:06:50	"s" dataset summary:
# 	9088534 rows x 9 columns | 5340.69 MB approx memory usage
#       col_name                 col_type  null_count  nunique
# 0         uuid            <class 'str'>           0  9088534
# 1        phone            <class 'str'>           0  3398850
# 2       gender            <class 'str'>           0        6
# 3          dob            <class 'str'>           0    36751
# 4     language            <class 'str'>           0       17
# 5        email            <class 'str'>           0  3258713
# 6  report_type            <class 'str'>           0       78
# 7       device            <class 'str'>           0        5
# 8     log_date  <class 'datetime.date'>           0      827
#
# "s" dataset head:
#        uuid   phone  gender       dob language email report_type  device    log_date
# 0  10058150     145    Male  00000000      TAM     0       LS-MT  mobile  2019-02-26
# 1         0     145    Male  00000000      TAM     0       LS-MT  mobile  2019-02-26
# 2         1     145    Male  00000000      TAM     0       LS-MT  mobile  2019-02-26
# 3  10058153  607734  Female  00000000      TEL     1       LS-MP  mobile  2019-02-26
# 4        26  607735  Female  00000000      TAM     2       LS-MT  mobile  2019-02-26
#
# 2021-09-18 22:06:50	Processing stage 2: consolidating "b_3m" dataset by date and uuid ...
# 2021-09-18 22:07:11		Consolidating sum_beacon_value ...
# 2021-09-18 22:08:54		Consolidating nunique_beacon_type ...
# 2021-09-18 22:11:07		Consolidating count_user_stay ...
# 2021-09-18 22:12:53		Consolidating count_pay_attempt ...
# 2021-09-18 22:16:14		Consolidating count_buy_click ...
#
# 2021-09-18 22:19:37	"b_3m_cb_date_uuid" dataset summary:
# 	1811593 rows x 7 columns | 295.49 MB approx memory usage
#               col_name               col_type  null_count  nunique
# 0                 date          <class 'str'>           0       92
# 1                 uuid          <class 'str'>           0  1794005
# 2     sum_beacon_value  <class 'numpy.int64'>           0     1563
# 3  nunique_beacon_type  <class 'numpy.int64'>           0       13
# 4      count_user_stay  <class 'numpy.int64'>           0      163
# 5    count_pay_attempt  <class 'numpy.int64'>           0       42
# 6      count_buy_click  <class 'numpy.int64'>           0       78
#
# "b_3m_cb_date_uuid" dataset head:
#          date     uuid  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click
# 0  2021-05-01  1446394               355                   13               28                141              152
# 1  2021-05-01  4167303                 3                    1                0                  3                0
# 2  2021-05-01  6005417                 1                    1                0                  1                0
# 3  2021-05-01  7017557                 1                    1                1                  0                0
# 4  2021-05-01  7192621                 1                    1                0                  1                0
#
# 2021-09-18 22:19:37	Processing stage 3: merging "s" with "b_3m_cb_date_uuid" ...
#
# 2021-09-18 22:20:24	"bs_merged" dataset summary:
# 	1604430 rows x 14 columns | 1107.75 MB approx memory usage
#                col_name               col_type  null_count  nunique
# 0                  date          <class 'str'>           0       92
# 1                  uuid          <class 'str'>           0  1597911
# 2      sum_beacon_value  <class 'numpy.int64'>           0     1556
# 3   nunique_beacon_type  <class 'numpy.int64'>           0       13
# 4       count_user_stay  <class 'numpy.int64'>           0      163
# 5     count_pay_attempt  <class 'numpy.int64'>           0       34
# 6       count_buy_click  <class 'numpy.int64'>           0       34
# 7                 phone          <class 'str'>           0   835655
# 8                gender          <class 'str'>           0        6
# 9                   dob          <class 'str'>           0    30252
# 10             language          <class 'str'>           0       14
# 11                email          <class 'str'>           0   824412
# 12          report_type          <class 'str'>           0       68
# 13               device          <class 'str'>           0        5
#
# "bs_merged" dataset head:
#          date     uuid  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click   phone gender         dob language   email report_type  device
# 0  2021-05-01  1446394               355                   13               28                141              152  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 1  2021-05-02  1446394                96                    5                0                 33               63  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 2  2021-05-03  1446394               157                   11               17                 33               97  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 3  2021-05-04  1446394               229                   11               28                 32              139  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 4  2021-05-05  1446394               290                   12               40                 41              140  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile

######################

# /home/ngkpg/anaconda3/envs/pyconda37/bin/python3.7 /home/ngkpg/Documents/Packt_GP/GP1/code/data_prep_1.py skip
# INFO: Pandarallel will run on 4 workers.
# INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.
#
# 2021-09-18 23:15:26	Reading dataset: bs_merged_3m.csv ...
#
# 2021-09-18 23:15:28	"bs_merged" dataset summary:
# 	1604430 rows x 14 columns | 680.17 MB approx memory usage
#                col_name               col_type  null_count  nunique
# 0                  date          <class 'str'>           0       92
# 1                  uuid  <class 'numpy.int64'>           0  1597911
# 2      sum_beacon_value  <class 'numpy.int64'>           0     1556
# 3   nunique_beacon_type  <class 'numpy.int64'>           0       13
# 4       count_user_stay  <class 'numpy.int64'>           0      163
# 5     count_pay_attempt  <class 'numpy.int64'>           0       34
# 6       count_buy_click  <class 'numpy.int64'>           0       34
# 7                 phone  <class 'numpy.int64'>           0   835655
# 8                gender          <class 'str'>           0        6
# 9                   dob          <class 'str'>           0    30252
# 10             language          <class 'str'>           0       14
# 11                email  <class 'numpy.int64'>           0   824412
# 12          report_type          <class 'str'>           0       68
# 13               device          <class 'str'>           0        5
#
# "bs_merged" dataset head:
#          date     uuid  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click   phone gender         dob language   email report_type  device
# 0  2021-05-01  1446394               355                   13               28                141              152  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 1  2021-05-02  1446394                96                    5                0                 33               63  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 2  2021-05-03  1446394               157                   11               17                 33               97  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 3  2021-05-04  1446394               229                   11               28                 32              139  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
# 4  2021-05-05  1446394               290                   12               40                 41              140  395287   Male  1996-06-14      HIN  524282       LS-MT  mobile
#
# 2021-09-18 23:15:28	Processing stage 4: consolidating "bs_merged" dataset by date and email ...
# 2021-09-18 23:15:37		Consolidating count_sessions ...
# 2021-09-18 23:16:15		Consolidating sum_beacon_value ...
# 2021-09-18 23:17:10		Consolidating nunique_beacon_type ...
# 2021-09-18 23:18:05		Consolidating count_user_stay ...
# 2021-09-18 23:19:01		Consolidating count_pay_attempt ...
# 2021-09-18 23:19:56		Consolidating count_buy_click ...
# 2021-09-18 23:20:53		Consolidating nunique_gender ...
# 2021-09-18 23:22:09		Consolidating nunique_dob ...
# 2021-09-18 23:23:22		Consolidating nunique_language ...
# 2021-09-18 23:24:35		Consolidating nunique_report_type ...
# 2021-09-18 23:25:48		Consolidating nunique_device ...
#
# 2021-09-18 23:27:04	"bs_merged_cb_date_email" dataset summary:
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
# "bs_merged_cb_date_email" dataset head:
#          date  email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device
# 0  2021-05-01    125               3                30                    3               12                  0                0               2            2                 1                    1               1
# 1  2021-05-01    141               5                39                    5               16                  0                0               2            5                 4                    2               2
# 2  2021-05-01    195               1                10                    1                4                  0                0               1            1                 1                    1               1
# 3  2021-05-01    645               1                10                    1                4                  0                0               1            1                 1                    1               1
# 4  2021-05-01    798               1                 3                    1                2                  0                0               1            1                 1                    1               1
