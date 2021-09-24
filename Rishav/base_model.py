# This script creates the final base model from our list of tuned models
# and dumps it to a joblib file

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
print("Standard scaling the Data ...")
scaler = StandardScaler()
X_scaled_s_f1 = scaler.fit_transform(X[feature_set_1])

print("Creating the SDC model with tuned hyperparameters ...")
final_base_model = SDC(random_state=23, max_iter=3000, loss='log', alpha=0.00001, class_weight='balanced', penalty='l1')

# Fitting the model
print("Fitting the model ...")
final_base_model.fit(X_scaled_s_f1, y)
print("Dumbping begins ...")
joblib.dump(final_base_model, 'SDC_fs1_s_jlib')
print("Model dumped ...")

###############################################################################


# 2021-09-24 05:17:30     Reading dataset: base_data_resampled_tomek.csv ...

# Splitting into features and lables ...
# Creating the necessary feature set ...
# Standard scaling the Data ...
# Creating the SDC model with tuned hyperparameters ...
# Fitting the model ...
# Dumbping begins ...
# Model dumped ...