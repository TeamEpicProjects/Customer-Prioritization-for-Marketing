import pandas as pd
import os
from collections import Counter
import datetime
from imblearn.under_sampling import TomekLinks

base_path = os.path.dirname(os.path.realpath(__file__))

print('\n{}\tReading data with outlier removed ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
in_filename = 'base_data_dev_3m_or.csv'
df_final_or = pd.read_csv(os.path.join(base_path, in_filename))
print('\n{}\t"final_or" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns '.format(df_final_or.shape[0], df_final_or.shape[1]))
print('\n"final_or" dataset head:')
print(df_final_or.head())

##########################

print("\n{}\tRandom Resampling begins...\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print("\nExtracting rows where the conversion_status is 0 ...")
df_final_status0 = df_final_or[df_final_or['conversion_status'] == 0]           # Extracting the entries where conversion_status is 0 for sampling
print(f"\n\t{df_final_status0.shape[0]} rows x {df_final_status0.shape[1]} columns")
print(f"The number of entries with conversion status 1 is {df_final_or.shape[0] - df_final_status0.shape[0]}") # Calculates the rows where the conversion_status is 1.
print("Choosing 15% extra sampled data ...")
n = int(1.15*(df_final_or.shape[0] - df_final_status0.shape[0]))                # Calculates the number of entries that would make n 15% more than the conversion_status 1 entries
print(f"Choosing a sample with {n} entries...")
df_final_status0_sample = df_final_status0.sample(n=n, random_state=23, ignore_index=True)
print("Sample with status 0 chosen.")
print(f" {df_final_status0_sample.shape[0]} rows x {df_final_status0_sample.shape[1]} columns")
df_final_status1 = df_final_or[df_final_or['conversion_status'] == 1]           # Extracting the rows where conversion_status is 1
print("\nConcatenating the two dataframes df_final_status1 and df_final_status0_sample...")
final_resampled_1 = pd.concat([df_final_status0_sample, df_final_status1], ignore_index=True) # Concatenating the two dataframes into one
print("Dataset summary...")
print(f"Total rows that should be: {df_final_status0_sample.shape[0] + df_final_status1.shape[0]}")   # Calculates the total rows that should be present in our concatenated dataset for manual check
print(f"Total rows there are: {final_resampled_1.shape[0]}")
print(f"Toal columns {final_resampled_1.shape[1]}")
print("{}\tRandom undersampling done!".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

###########################

print("Seperating Features and Labels for sampling ...")
X, y = final_resampled_1.drop('conversion_status', axis=1), final_resampled_1['conversion_status']
print('\n{}\Tomek Link resampling begins ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print(f"\nBefore TL resampling, value_counts: {Counter(y)}")
t = TomekLinks(n_jobs=-1)
X_t, y_t = t.fit_resample(X, y)
print(f"\nAfer TL resampling, value_counts: {Counter(y_t)}")
X_t['conversion_status'] = y_t.values                      # Adding our target back to the dataframe
final_resampled_t1 = X_t.copy()
print('\n{}\Tomek Link resampling ends ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print("Saving resampled_data_tomek.csv ...")
final_resampled_t1.to_csv('resampled_data_tomek.csv', encoding='utf-8', index=False)

###############################



# 2021-09-20 22:10:00     Reading data with outlier removed ...

# 2021-09-20 22:10:00     "final_or" dataset summary:
#         788332 rows x 14 columns

# "final_or" dataset head:
#    count_sessions  sum_beacon_value  ...  profile_submit_count  transactions_amount
# 0               1                10  ...                     2                  0.0
# 1               1                10  ...                     1                  0.0
# 2               1               106  ...                     2                  0.0
# 3               1                14  ...                     8                  0.0
# 4               1                 2  ...                     8                  0.0

# [5 rows x 14 columns]

# 2021-09-20 22:10:00     Random Resampling begins...


# Extracting rows where the conversion_status is 0 ...

#         776098 rows x 14 columns
# The number of entries with conversion status 1 is 12234
# Choosing 15% extra sampled data ...
# Choosing a sample with 14069 entries...
# Sample with status 0 chosen.
#  14069 rows x 14 columns

# Concatenating the two dataframes df_final_status1 and df_final_status0_sample...
# Dataset summary...
# Total rows that should be: 26303
# Total rows there are: 26303
# Toal columns 14
# 2021-09-20 22:10:01     Random undersampling done!
# Seperating Features and Labels for sampling ...

# 2021-09-20 22:10:01\Tomek Link resampling begins ...

# Before TL resampling, value_counts: Counter({0: 14069, 1: 12234})

# Afer TL resampling, value_counts: Counter({0: 13939, 1: 12234})

# Afer TL resampling, value_counts: Counter({0: 13939, 1: 12234})

# 2021-09-20 22:10:02\Tomek Link resampling ends ...
# Saving resampled_data_tomek.csv ...

