# This script creates the final base model from our list of tuned models
# and dumps it to a joblib file

import pandas as pd
import os
import datetime
from sklearn.linear_model import LogisticRegression as LRC
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
print("Creating the model ...")
final_base_model = LRC(random_state=23, max_iter=3000, penalty='l2', C=0.001, class_weight='balanced', solver='newton-cg')
feature_set_4 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_report_type', 'nunique_device', 'transactions_amount']

# Fitting the model
print("Fitting the model ...")
final_base_model.fit(X[feature_set_4], y)
joblib.dump(final_base_model, 'LR_fs4_us_jlib')
print("Model dumped ...")

###############################################################################

# 2021-09-23 10:45:46     Reading dataset: base_data_resampled_tomek.csv ...

# Splitting into features and lables ...
# Creating the model ...
# Fitting the model ...
# Model dumped ...