# This script creates three base models for testing from our list of tuned models
# The three models are:
# SDC2 feature_set_4 std scaled tuned - 6 features
# SDC2 feature_set_1 std scaled tuned - 7 features
# SDC3 feature_set_5 std scaled tuned - 4 features

import pandas as pd
import os
import datetime
from sklearn.linear_model import SGDClassifier as SDC
from sklearn.preprocessing import StandardScaler 
import joblib

# Reading the dataset base_data_resampled_tomek.csv
base_path = os.path.dirname(os.path.realpath(__file__))
in_file_name = "base_data_resampled_tomek.csv"
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name))
df_test = pd.read_csv(os.path.join(base_path, in_file_name))

# Seperating the features and labels
print("\nSplitting into features and lables ...")
X = df_test.drop(columns=['conversion_status', 'email', 'date'], axis=1)
y = df_test['conversion_status']

# Creating the selected model from tuned_model_selection
print("Creating the necessary feature set ...")

feature_set_1 = ['transactions_amount', 'count_pay_attempt', 'nunique_beacon_type',
                 'count_user_stay', 'count_buy_click', 'profile_submit_count',
                 'sum_beacon_value']

feature_set_4 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_report_type', 'nunique_device', 'transactions_amount']

feature_set_5 = ['count_pay_attempt', 'count_buy_click',
                 'nunique_report_type', 'profile_submit_count']

print("Standard scaling the Data ...")
scaler = StandardScaler()
X_scaled_s_f1 = scaler.fit_transform(X[feature_set_1])
X_scaled_s_f4 = scaler.fit_transform(X[feature_set_4])
X_scaled_s_f5 = scaler.fit_transform(X[feature_set_5])

print("Creating the SDC model with tuned hyperparameters ...")

final_base_model_t2 = SDC(random_state=23, max_iter=3000, loss='log', alpha=0.00001, penalty='elasticnet')
final_base_model_t3 = SDC(random_state=23, max_iter=3000, loss='log', alpha=0.01, class_weight='balanced', penalty='l2')


# Fitting and dumping the 3 models

print("Dumping 1 ...")
final_base_model_t2.fit(X_scaled_s_f1, y)
joblib.dump(final_base_model_t2, 'SDC_f1_s_jlib.pkl')

print("Dumping 2 ...")
final_base_model_t2.fit(X_scaled_s_f4, y)
joblib.dump(final_base_model_t2, 'SDC_f4_s_jlib.pkl')


print("Dumping 4 ...")
final_base_model_t3.fit(X_scaled_s_f5, y)
joblib.dump(final_base_model_t3, 'SDC_f5_s_t3_jlib.pkl')

print("Model dumped ...")

###############################################################################

# 2021-09-27 09:39:53     Reading dataset: base_data_resampled_tomek.csv ...

# Splitting into features and lables ...
# Creating the necessary feature set ...
# Standard scaling the Data ...
# Creating the SDC model with tuned hyperparameters ...
# Dumping 1 ...
# Dumping 2 ...
# Dumping 3 ...
# Model dumped ...
