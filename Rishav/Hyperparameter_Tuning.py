# This script will tune the hyperparameters of the following models
# Logistic Regression feature_set_1, feature_set_4 non-scaled
# Logistic Regression feature_set_1, feature_set_4 Standard scaled
# Stochastic Gradient Descent feature_set_1, feature_set_3, feature_set_4 non-scaled
# Stochastic Gradient Descent feature_set_1, feature_set_3, feature_set_4 Standard scaled

import os
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LRC, SGDClassifier as SDC
from sklearn.preprocessing import StandardScaler

# Reading the dataset base_data_resampled_tomek.csv
base_path = os.path.dirname(os.path.realpath(__file__))
in_file_name = "base_data_resampled_tomek.csv"
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name))
df_test = pd.read_csv(os.path.join(base_path, in_file_name))

# Seperating the features and labels
print("\nSplitting into features and lables ...")
X = df_test.drop(columns=['conversion_status', 'email', 'date'])
y = df_test['conversion_status']

# Listing the feature sets selected from the training script
print("Creating feature sets ...")
feature_set_1 = ['transactions_amount', 'count_pay_attempt', 'nunique_beacon_type',
                 'count_user_stay', 'count_buy_click', 'profile_submit_count',
                 'sum_beacon_value']
feature_set_3 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_dob', 'nunique_language', 'nunique_report_type',
                 'nunique_device', 'transactions_amount']
feature_set_4 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_report_type', 'nunique_device', 'transactions_amount']

# Metrics to be used for cross validation
metrics = ['balanced_accuracy', 'recall']
verbose_ = 5

# Scaling the X's
print("Standard Scaling data ...")
scaler = StandardScaler()
X_scaled_s_f1 = scaler.fit_transform(X[feature_set_1])
X_scaled_s_f3 = scaler.fit_transform(X[feature_set_3])
X_scaled_s_f4 = scaler.fit_transform(X[feature_set_4])

#################################################################################

# LOGISTIC REGRESSION HYPERPARAMETER TUNING
# Creating parameter grids for Logistic Regression
print("Creating parameter grids ...")
param_grid_lr1 = {'C': [0.001, 0.01, 0.1, 1],
                  'class_weight': [None, 'balanced'],
                  'solver': ['newton-cg', 'lbfgs']
                  }

param_grid_lr2 = {'C': [0.01, 0.1, 1],
                  'class_weight': [None, 'balanced'],
                  'penalty': ['l1', 'l2']}


# Cross validating both parameter grids with Grid Search CV
lr_tuning1 = GridSearchCV(LRC(random_state=23, max_iter=2000, penalty='l2'),
                         param_grid=param_grid_lr1, n_jobs=-1, cv=5,
                         verbose=verbose_, return_train_score=True,
                         refit='balanced_accuracy', scoring=metrics)

lr_tuning2 = GridSearchCV(LRC(random_state=23, max_iter=4000, solver='saga'), param_grid=param_grid_lr2,
                         n_jobs=-1, refit='balanced_accuracy', cv=5,
                         verbose=verbose_, return_train_score=True, scoring=metrics)

# Fitting models on lr_tuning1 and lr_tuning2

# Fitting feature_set_1 and feature_set_4 on lr_tuning1 and getting the best parameters
print("Fitting lr_tuning1 with feature_set_1 ...")
lr_tuning1.fit(X[feature_set_1], y)
lr1_fs1_best = lr_tuning1.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning1.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_recall'])))
print("Fitting lr_tuning_1 with feature_set_4 ...")
lr_tuning1.fit(X[feature_set_4], y)
lr1_fs4_best = lr_tuning1.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning1.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_recall'])))


# Fitting feature_set_1 and feature_set_4 standard scaled on lr_tuning1 and getting the best parameters
print("Fitting lr_tuning1 on feature_set_1 standard scaled ...")
lr_tuning1.fit(X_scaled_s_f1, y)
lr1_fs1_scaled_best = lr_tuning1.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning1.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_recall'])))
print("Fitting lr_tuned_1 on feature_set_4 scaled ...")
lr_tuning1.fit(X_scaled_s_f4, y)
lr1_fs4_scaled_best = lr_tuning1.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning1.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning1.cv_results_['mean_train_recall'])))

# Fitting feature_set_1 and feature_set_4 on lr_tuning2 and getting the best parameters
print("Fitting lr_tuning2 on feature_set_1 ...")
lr_tuning2.fit(X[feature_set_1], y)
lr2_fs1_best = -1  # arbitrary number temporary
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning2.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_recall'])))
print("Fitting lr_tuning2 with feature_set_4 ...")
lr_tuning2.fit(X[feature_set_4], y)
lr2_fs4_best = -1 
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning2.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_recall'])))


# Fitting feature_set_1 and feature_set_4 standard scaled on lr_tuning2 and getting the best params
print("Fitting lr_tuning2 on feature set1 standard scaled ...")
lr_tuning2.fit(X_scaled_s_f1, y)
lr2_fs1_scaled_best = lr_tuning2.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning2.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_recall'])))

print("Fitting lr_tuning_2 on feature_set_4 standard scaled ...")
lr_tuning2.fit(X_scaled_s_f4, y)
lr2_fs4_scaled_best = lr_tuning2.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(lr_tuning2.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_test_recall']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(lr_tuning2.cv_results_['mean_train_recall'])))


################################################################################

# STOCHASTIC GRADIENT DESCENT HYPERPARAMETER TUNING
# Creating parameter grids for Stochastic Gradient Descent 

param_grid_sdc = {'penalty': ['l2', 'l1', 'elasticnet'],
                  'alpha': [0.00001, 0.0001, 0.01, 0.1],
                  'class_weight': [None, 'balanced']
                  }

# Cross validating parameter grid with Grid Search CV
sdc_tuning = GridSearchCV(SDC(random_state=23, max_iter=2000, loss='log'),
                          param_grid=param_grid_sdc, cv=5, n_jobs=-1,
                          verbose=verbose_, refit='balanced_accuracy',
                          return_train_score=True, scoring=metrics)

# Fitting feature_set_1, feature_set_3 and feature_set_4 on sdc_tuning and getting the best parameters
print("Fitting sdc_tuning on feature_set_1 ...")
sdc_tuning.fit(X[feature_set_1], y)
sdc_fs1_best = sdc_tuning.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(sdc_tuning.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_test_recall']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_recall'])))
print("Fitting sdc_tuning on feature_set_3 ...")
sdc_tuning.fit(X[feature_set_3], y)
sdc_fs3_best = sdc_tuning.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(sdc_tuning.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_test_recall']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_recall'])))

print("Fitting sdc_tuning on feature_set_4 ...")
sdc_tuning.fit(X[feature_set_4], y)
sdc_fs4_best = sdc_tuning.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(sdc_tuning.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_test_recall']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_recall'])))

# Fitting feature_set_1, feature_set_3 and feature_set_4 on sdc_tuning and getting the best parameters
print("Fitting sdc_tuned on feature_set_1 standard scaled ...")
sdc_tuning.fit(X_scaled_s_f1, y)
sdc_fs1_scaled_best = sdc_tuning.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(sdc_tuning.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_test_recall']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_recall'])))

# Fit with feature set 3 scaled
print("Fitting sdc_tuned on feature_set_3 standard scaled ...")
sdc_tuning.fit(X_scaled_s_f3, y)
sdc_fs3_scaled_best = sdc_tuning.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(sdc_tuning.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_test_recall']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_recall'])))

# Fit with feature set 4 scaled
print("Fitting sdc_tuned on feature_set_4 standard scaled ...")
sdc_tuning.fit(X_scaled_s_f4, y)
sdc_fs4_scaled_best = sdc_tuning.best_params_
print("Test_BAC: {}\tTest_RCC: {}\t Train_BAC: {}\t Train_RCC: {}".format(np.mean(sdc_tuning.cv_results_['mean_test_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_test_recall']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_balanced_accuracy']),
                                                                          np.mean(sdc_tuning.cv_results_['mean_train_recall'])))

print("\nBest hyperparameters for LOGISTIC REGRESSION models: -")
logit_best_params = [['LR1 feature_set_1', lr1_fs1_best],['LR1 feature_set_4', lr1_fs4_best],
                     ['LR1 feature_set_1 StdScaled', lr1_fs1_scaled_best],
                     ['LR1 feature_set_4 StdScaled', lr1_fs4_scaled_best],
                     ['LR2 feature_set_1', lr2_fs1_best], ['LR2 feature_set_4', lr2_fs4_best],
                     ['LR2 feature_set_1 StdScaled', lr2_fs1_scaled_best],
                     ['LR2 feature_set_4 StdScaled', lr2_fs4_scaled_best]]

df_logit_best_params = pd.DataFrame(logit_best_params, columns=['model', 'best hyperparameters'])
print(df_logit_best_params.to_string())

print("\nBest hyperparameters for STOCHASTIC GRADIENT DESCENT: - ")
sgd_best_params = [['SGD feature_set_1', sdc_fs1_best], ['SGD feature_set_3', sdc_fs3_best],
                   ['SGD feature_set_4', sdc_fs4_best], ['SGD feature_set_1 StdScaled', sdc_fs1_scaled_best],
                   ['SGD feature_set_3 StdScaled', sdc_fs3_scaled_best],['SGD feature_set_4 StdScaled', sdc_fs4_scaled_best]]

df_sgd_best_params = pd.DataFrame(sgd_best_params, columns=['model', 'best hyperparameters'])
print(df_sgd_best_params.to_string())
print("Complete!")

################################################################################
