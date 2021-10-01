# This script takes the prepared data, removes outliers and resamples to create a new dataset
#   Inputs: base_data_dev_3m.csv
#   Output: base_data_resampled_tomek.csv

import pandas as pd
import os
from collections import Counter
import datetime
from imblearn.under_sampling import TomekLinks

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


# Reading the dataset base_data_dev_3m and printing it's basic summary
print('\n{}\tReading base_data_dev_3m ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = 'base_data_dev_3m.csv'
df_final = pd.read_csv(os.path.join(base_path, in_filename))
df_final_info = df_info(df_final)
print('\n{}\t"base_data_dev_3m" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_final.shape[0], df_final.shape[1], df_final_info[1]))
print(df_final_info[0].to_string())
print('\n"base_data_dev_3m_or" dataset head:')
print(df_final.head().to_string())

##########################################################

# Removing outliers based on the certain thresholds values which,
# due to lack of time were obtained by consulting domain experts
df_final_or = df_final[(df_final.count_user_stay <= 50) & (df_final.count_pay_attempt <= 10)
                       & (df_final.count_buy_click <= 25) & (df_final.nunique_gender <= 2)
                       & (df_final.nunique_dob <= 5) & (df_final.nunique_report_type <= 2)
                       & (df_final.nunique_language <= 2)]
# Storing the outlier removed file into a csv
df_final_or.to_csv('base_data_dev_3m_or.csv', encoding='utf-8', index=False)

###########################################################

# Reading the dataset base_data_dev_3m_or and printing its basic summary
print('\n{}\tReading data with outliers removed ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = 'base_data_dev_3m_or.csv'
df_final_or = pd.read_csv(os.path.join(base_path, in_filename))
print("Removing date column. To be appended later ...")
df_final_or_info = df_info(df_final_or)
print('\n{}\t"base_data_dev_3m_or" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_final_or.shape[0], df_final_or.shape[1], df_final_or_info[1]))
print(df_final_or_info[0].to_string())
print('\n"base_data_dev_3m_or" dataset head:')
print(df_final_or.head().to_string())

############################################################

# Random resampling the dataset base_data_dev_3m_or
print("\n{}\tRandom Resampling of base_data_dev_3m_or begins...\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print("\nExtracting rows where the conversion_status is 0 ...")

# Extracting the entries where conversion_status is 0 
df_final_status0 = df_final_or[df_final_or['conversion_status'] == 0]          
print("\nDimension of extracted rows ...")
print(f"\n\t{df_final_status0.shape[0]} rows x {df_final_status0.shape[1]} columns")
print(f"\nThe number of entries with conversion status 1 is {df_final_or.shape[0] - df_final_status0.shape[0]}") 
# Calculates the number of entries that would make n 15% more than the conversion_status 1 entries
n = int(1.15*(df_final_or.shape[0] - df_final_status0.shape[0]))               
print(f"Choosing a sample with 15% extra, total {n} entries...")
# Choosing a random sample and printing it's basic summary
df_final_status0_sample = df_final_status0.sample(n=n, random_state=23, ignore_index=True)
print("Sample with conversion_status as 0 chosen ...")
print("\nDimension of the sampled data with conversion_status as 0:-")
print(f"{df_final_status0_sample.shape[0]} rows x {df_final_status0_sample.shape[1]} columns")
print(df_info(df_final_status0_sample))

print("\nSelecting all the rows where conversion_status is 1 ...")
# Finding the rows where conversion status is 1
df_final_status1 = df_final_or[df_final_or['conversion_status'] == 1]           # Extracting the rows where conversion_status is 1
print("\nConcatenating the two dataframes df_final_status1 and df_final_status0_sample...")
# Concatenating the two data frames status0 sampled and status1 complete
final_resampled_1 = pd.concat([df_final_status0_sample, df_final_status1], ignore_index=True)  # Concatenating the two dataframes into one
print(f"Total rows that should be: {df_final_status0_sample.shape[0] + df_final_status1.shape[0]}")   # Calculates the total rows that should be present in our concatenated dataset for manual check
print(f"Total rows there are: {final_resampled_1.shape[0]}")
print("It matches!")
final_resampled_1.reset_index(inplace=True, drop=True)

# Printing the basic summary of the randomly sampled dataset.
print("\nDataset summary...")
final_resampled_1_info = df_info(final_resampled_1)
print('\n{}\t"final_resampled_1_info" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(final_resampled_1.shape[0], final_resampled_1.shape[1], df_final_or_info[1]))
print(final_resampled_1_info[0].to_string())
print('\n"final_resampled_1" dataset head:')
print(final_resampled_1.head().to_string())
print("{}\tRandom undersampling done!".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


##########################################################

print("Separating Features and Labels for sampling ...")
print("Temporarily converting dates to integer for Tomek sampling ...")
# Preparing the date column for Tomek Sampling by converting it into integer
final_resampled_1['date'] = final_resampled_1['date'].apply(lambda x: int(''.join(x.split('-'))))
# Separating the features and targets
X, y = final_resampled_1.drop('conversion_status', axis=1), final_resampled_1['conversion_status']
print('\n{}Tomek Link resampling begins ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print(f"\nBefore TL resampling, value_counts: {Counter(y)}")
t = TomekLinks(n_jobs=-1)
X_t, y_t = t.fit_resample(X, y)
print(f"\nAfter TL resampling, value_counts: {Counter(y_t)}")
X_t['conversion_status'] = y_t.values 
# Adding the target that we separated earlier back to the dataframe
final_resampled_t1 = X_t.copy()
print('\n{}Tomek Link resampling ends ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# Reinstating the date column in the same format as a string
final_resampled_t1['date'] = final_resampled_t1['date'].apply(lambda x: f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:]}")
final_resampled_t1.to_csv('base_data_resampled_tomek.csv', encoding='utf-8', index=False)
print("Saving resampled_data_tomek.csv ...")

# Printing the basic summary of the final undersampled dataset
print("\nSummary: ")
print(df_info(final_resampled_t1))


#############################################################


# 2021-09-21 15:09:53     Reading base_data_dev_3m ...

# 2021-09-21 15:09:54     "base_data_dev_3m" dataset summary:
#         798162 rows x 16 columns | 142.34 MB approx memory usage
#                 col_name                 col_type  null_count  nunique
# 0                   date            <class 'str'>           0       92
# 1                  email    <class 'numpy.int64'>           0   643967
# 2         count_sessions    <class 'numpy.int64'>           0       52
# 3       sum_beacon_value    <class 'numpy.int64'>           0     2257
# 4    nunique_beacon_type    <class 'numpy.int64'>           0       58
# 5        count_user_stay    <class 'numpy.int64'>           0      228
# 6      count_pay_attempt    <class 'numpy.int64'>           0       44
# 7        count_buy_click    <class 'numpy.int64'>           0       40
# 8         nunique_gender    <class 'numpy.int64'>           0        3
# 9            nunique_dob    <class 'numpy.int64'>           0       35
# 10      nunique_language    <class 'numpy.int64'>           0        8
# 11   nunique_report_type    <class 'numpy.int64'>           0       12
# 12        nunique_device    <class 'numpy.int64'>           0        4
# 13     conversion_status    <class 'numpy.int64'>           0        2
# 14  profile_submit_count    <class 'numpy.int64'>           0      360
# 15   transactions_amount  <class 'numpy.float64'>           0     1288

# "base_data_dev_3m_or" dataset head:
#          date    email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device  conversion_status  profile_submit_count  transactions_amount
# 0  2021-07-29  3237641               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     2                  0.0
# 1  2021-05-15  2637791               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     1                  0.0
# 2  2021-06-11  2807521               1               106                    2               14                  0                1               1            1                 1                    1               1                  0                     2                  0.0
# 3  2021-06-02   832889               1                14                    1                5                  0                0               1            1                 1                    1               1                  0                     8                  0.0
# 4  2021-07-05  2624659               1                 2                    2                1                  0                0               1            1                 1                    1               1                  0                     8                  0.0

# 2021-09-21 15:09:57     Reading data with outliers removed ...
# Removing date column. To be appended later ...

# 2021-09-21 15:09:58     "base_data_dev_3m_or" dataset summary:
#         788332 rows x 16 columns | 140.59 MB approx memory usage
#                 col_name                 col_type  null_count  nunique
# 0                   date            <class 'str'>           0       92
# 1                  email    <class 'numpy.int64'>           0   639763
# 2         count_sessions    <class 'numpy.int64'>           0       26
# 3       sum_beacon_value    <class 'numpy.int64'>           0      980
# 4    nunique_beacon_type    <class 'numpy.int64'>           0       27
# 5        count_user_stay    <class 'numpy.int64'>           0       51
# 6      count_pay_attempt    <class 'numpy.int64'>           0       11
# 7        count_buy_click    <class 'numpy.int64'>           0       18
# 8         nunique_gender    <class 'numpy.int64'>           0        2
# 9            nunique_dob    <class 'numpy.int64'>           0        5
# 10      nunique_language    <class 'numpy.int64'>           0        2
# 11   nunique_report_type    <class 'numpy.int64'>           0        2
# 12        nunique_device    <class 'numpy.int64'>           0        3
# 13     conversion_status    <class 'numpy.int64'>           0        2
# 14  profile_submit_count    <class 'numpy.int64'>           0      360
# 15   transactions_amount  <class 'numpy.float64'>           0     1259

# "base_data_dev_3m_or" dataset head:
#          date    email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device  conversion_status  profile_submit_count  transactions_amount
# 0  2021-07-29  3237641               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     2                  0.0
# 1  2021-05-15  2637791               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     1                  0.0
# 2  2021-06-11  2807521               1               106                    2               14                  0                1               1            1                 1                    1               1                  0                     2                  0.0
# 3  2021-06-02   832889               1                14                    1                5                  0                0               1            1                 1                    1               1                  0                     8                  0.0
# 4  2021-07-05  2624659               1                 2                    2                1                  0                0               1            1                 1                    1               1                  0                     8                  0.0

# 2021-09-21 15:09:58     Random Resampling of base_data_dev_3m_or begins...


# Extracting rows where the conversion_status is 0 ...

# Dimension of extracted rows ...

#         776098 rows x 16 columns

# The number of entries with conversion status 1 is 12234
# Choosing a sample with 15% extra, total 14069 entries...
# Sample with conversion_status as 0 chosen ...

# Dimension of the sampled data with conversion_status as 0:-
# 14069 rows x 16 columns
# (                col_name                 col_type  null_count  nunique
# 0                   date            <class 'str'>           0       92
# 1                  email    <class 'numpy.int64'>           0    13933
# 2         count_sessions    <class 'numpy.int64'>           0       15
# 3       sum_beacon_value    <class 'numpy.int64'>           0      237
# 4    nunique_beacon_type    <class 'numpy.int64'>           0       15
# 5        count_user_stay    <class 'numpy.int64'>           0       51
# 6      count_pay_attempt    <class 'numpy.int64'>           0        7
# 7        count_buy_click    <class 'numpy.int64'>           0        7
# 8         nunique_gender    <class 'numpy.int64'>           0        2
# 9            nunique_dob    <class 'numpy.int64'>           0        5
# 10      nunique_language    <class 'numpy.int64'>           0        2
# 11   nunique_report_type    <class 'numpy.int64'>           0        2
# 12        nunique_device    <class 'numpy.int64'>           0        2
# 13     conversion_status    <class 'numpy.int64'>           0        1
# 14  profile_submit_count    <class 'numpy.int64'>           0      221
# 15   transactions_amount  <class 'numpy.float64'>           0       91, 2.5109777450561523)

# Selecting all the rows where conversion_status is 1 ...

# Concatenating the two dataframes df_final_status1 and df_final_status0_sample...
# Total rows that should be: 26303
# Total rows there are: 26303
# It matches!

# Dataset summary...

# 2021-09-21 15:09:58     "final_resampled_1_info" dataset summary:
#         26303 rows x 16 columns | 140.59 MB approx memory usage
#                 col_name                 col_type  null_count  nunique
# 0                   date            <class 'str'>           0       92
# 1                  email    <class 'numpy.int64'>           0    24952
# 2         count_sessions    <class 'numpy.int64'>           0       18
# 3       sum_beacon_value    <class 'numpy.int64'>           0      450
# 4    nunique_beacon_type    <class 'numpy.int64'>           0       22
# 5        count_user_stay    <class 'numpy.int64'>           0       51
# 6      count_pay_attempt    <class 'numpy.int64'>           0       11
# 7        count_buy_click    <class 'numpy.int64'>           0       13
# 8         nunique_gender    <class 'numpy.int64'>           0        2
# 9            nunique_dob    <class 'numpy.int64'>           0        5
# 10      nunique_language    <class 'numpy.int64'>           0        2
# 11   nunique_report_type    <class 'numpy.int64'>           0        2
# 12        nunique_device    <class 'numpy.int64'>           0        3
# 13     conversion_status    <class 'numpy.int64'>           0        2
# 14  profile_submit_count    <class 'numpy.int64'>           0      246
# 15   transactions_amount  <class 'numpy.float64'>           0      981

# "final_resampled_1" dataset head:
#          date    email  count_sessions  sum_beacon_value  nunique_beacon_type  count_user_stay  count_pay_attempt  count_buy_click  nunique_gender  nunique_dob  nunique_language  nunique_report_type  nunique_device  conversion_status  profile_submit_count  transactions_amount
# 0  2021-07-14  3097420               1               352                    2               26                  0                0               1            1                 1                    1               1                  0                     2                  0.0
# 1  2021-06-18  2447087               6                60                    6               24                  0                0               2            3                 1                    1               1                  0                    63                  0.0
# 2  2021-06-20  1230062               1                 3                    1                2                  0                0               1            1                 1                    1               1                  0                     2                  0.0
# 3  2021-07-12  1275981               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     2                  0.0
# 4  2021-05-29  2697239               1                10                    1                4                  0                0               1            1                 1                    1               1                  0                     1                  0.0
# 2021-09-21 15:09:58     Random undersampling done!
# Separating Features and Labels for sampling ...
# Temporarily converting dates to integer for Tomek sampling ...

# 2021-09-21 15:09:58\Tomek Link resampling begins ...

# Before TL resampling, value_counts: Counter({0: 14069, 1: 12234})

# After TL resampling, value_counts: Counter({0: 13619, 1: 12234})

# 2021-09-21 15:09:58\Tomek Link resampling ends ...
# Saving resampled_data_tomek.csv ...

# Summary:
# (                col_name                 col_type  null_count  nunique
# 0                   date            <class 'str'>           0       92
# 1                  email    <class 'numpy.int64'>           0    24541
# 2         count_sessions    <class 'numpy.int64'>           0       17
# 3       sum_beacon_value    <class 'numpy.int64'>           0      447
# 4    nunique_beacon_type    <class 'numpy.int64'>           0       22
# 5        count_user_stay    <class 'numpy.int64'>           0       51
# 6      count_pay_attempt    <class 'numpy.int64'>           0       11
# 7        count_buy_click    <class 'numpy.int64'>           0       13
# 8         nunique_gender    <class 'numpy.int64'>           0        2
# 9            nunique_dob    <class 'numpy.int64'>           0        5
# 10      nunique_language    <class 'numpy.int64'>           0        2
# 11   nunique_report_type    <class 'numpy.int64'>           0        2
# 12        nunique_device    <class 'numpy.int64'>           0        3
# 13  profile_submit_count    <class 'numpy.int64'>           0      240
# 14   transactions_amount  <class 'numpy.float64'>           0      979
# 15     conversion_status    <class 'numpy.int64'>           0        2, 4.612502098083496)