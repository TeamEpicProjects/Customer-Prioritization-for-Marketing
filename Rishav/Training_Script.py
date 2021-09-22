# This script finds tests various Machine Learning Algorithms on our dataset

import pandas as pd
import os
import datetime
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LRC, SGDClassifier as SDC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier as GBC

# Reading the dataset base_data_resampled_tomek.csv
base_path = os.path.dirname(os.path.realpath(__file__))
in_file_name = "base_data_resampled_tomek.csv"
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name))
df_test = pd.read_csv(os.path.join(base_path, in_file_name))

# Seperating the features and labels
print("\nSplitting into features and lables ...")
X = df_test.drop(columns=['conversion_status', 'email', 'date'], axis=1)
y = df_test['conversion_status']

print("Building evaluation and training functions ...")

###############################################################################

# Creating an evaluation dictionary to store all the information
evaluation = {'model': [],
              'feature_count': [],
              'BAC_test': [],
              'Recall_test': [],
              'BAC_train': [],
              'Recall_train': [],
              'Fit_time': [],
              'Score_time': []
              } 

def model_evaluation(model_name, fc, c, bac_test, rcc_test, bac_train, rcc_train, ft, st):
    """
    To keep track of the models used and their
    respective characterisitics with evaluation scores.
    """
    evaluation['model'].append(model_name)
    evaluation['feature_count'].append(fc)
    evaluation['BAC_test'].append(bac_test)
    evaluation['Recall_test'].append(rcc_test)
    evaluation['BAC_train'].append(bac_train)
    evaluation['Recall_train'].append(rcc_train)
    evaluation['Fit_time'].append(ft)
    evaluation['Score_time'].append(st)
    
    df_eval = pd.DataFrame({'model_name': evaluation['model'],
                            'feature_count': evaluation['feature_count'],
                            'Balanced_Accuracy_test': evaluation['BAC_test'],
                            'Recall_test': evaluation['Recall_test'],
                            'Balanced_Accuracy_train': evaluation['BAC_train'],
                            'Recall_train': evaluation['Recall_train'],
                            'Fit_time': evaluation['Fit_time'],
                            'Score_time': evaluation['Score_time']
                            })
    
    return df_eval.sort_values(by='Balanced_Accuracy_test', ascending=False).round(3)


def fit_algorithm(algo_name, algorithm, X_, y_, cv=10):
    """
    Accepts data from the prepare data function and 
    return balanced accuracy scores of the predicted model.
    """
    model = cross_validate(algorithm, X_, y_, cv=cv, n_jobs=-1, return_train_score=True,
                           scoring=['balanced_accuracy', 'recall'])
        
    df = model_evaluation(algo_name, X_.shape[1], model['test_balanced_accuracy'].mean(),
                          model['test_recall'].mean(), model['train_balanced_accuracy'].mean(),
                          model['train_recall'].mean(), model['fit_time'].sum(), 
                          model['score_time'].sum())
    return df


###############################################################################

print("\nCreating models RandomForests, LogisticRegression, StochasticGradientDescent and GradientBoostClassifier...")
model_1 = RFC(random_state=23)
model_2 = LRC(random_state=23, max_iter=2500)
model_3 = SDC(random_state=23)
model_4 = GBC(random_state=23)

# Fitting the model using all the features
print("Fitting algorithms taking all features ...")
fit_algorithm("RF all features", model_1, X.to_numpy(), y)
fit_algorithm("LR all features", model_2, X.to_numpy(), y)
fit_algorithm("SGD all features", model_3, X.to_numpy(), y)

# Calculating the feature importances based on RandomForest
# Creating a dataframe of the features and their respective importances
rf_feature_imp = pd.DataFrame(abs((model_1.fit(X.to_numpy(), y).feature_importances_).round(4)).reshape(-1, 1), columns=['importance_magnitude'])
rf_feature_imp['features'] = pd.Series(X.iloc[:, rf_feature_imp.index.tolist()].columns.to_list())
# Setting the index of the feature_importance dataframe to 'features'
rf_feature_imp = rf_feature_imp.set_index('features').sort_values(by='importance_magnitude', ascending=False)
# Displaying the feature_importances of the columns of random forest
print(f"Feature importances from Random forests are:- \n{rf_feature_imp}")
print("Choosing the top 7 features to create feature_set_1 ...")
feature_set_1 = rf_feature_imp.index[0:7].to_list()
print(f"Feature set 1: {feature_set_1}")

# Fitting models using feature_set_1
print("\nFitting algorithms with feature_set_1 ...")
fit_algorithm("RF feature_set_1", model_1, X[feature_set_1], y)
fit_algorithm("LR feature_set_1", model_2, X[feature_set_1], y)
fit_algorithm("SGD feature_set_1", model_3, X[feature_set_1], y)
fit_algorithm("GB feature_set_1", model_4, X[feature_set_1], y)

# Calculating the feature importances based on LogisticRegression
# Creating a dataframe of the features and their respective importances
lr_feature_imp = pd.DataFrame(abs((model_2.fit(X.to_numpy(), y).coef_).round(4)).reshape(-1, 1), columns=['importance_magnitude'])
lr_feature_imp['features'] = pd.Series(X.iloc[:, lr_feature_imp.index.tolist()].columns.to_list())
lr_feature_imp = lr_feature_imp.set_index('features').sort_values(by='importance_magnitude', ascending=False)
print(f"Feature importances from Logistic Regression are:- \n{lr_feature_imp}")
# Choosing the top 7 features
print("Choosing the top 7 features from Logistic Regression model to create feature_set_2 ...")
feature_set_2 = lr_feature_imp.index[0:7].to_list()
print(f"Feature set 2: {feature_set_2}")

# Fitting models based on feature set 2
print("\nFitting algorithms with feature_set_2 ...")
fit_algorithm("RF feature_set_1", model_1, X[feature_set_2], y)
fit_algorithm("LR feature_set_1", model_2, X[feature_set_2], y)
fit_algorithm("SGD feature_set_1", model_3, X[feature_set_2], y)
fit_algorithm("GB feature_set_1", model_4, X[feature_set_2], y)

###############################################################################

# Scaling the data using max abs scaler to see if we can get more reliable models

transformer = MaxAbsScaler()
X_scaled_m_f1 = transformer.fit_transform(X[feature_set_1])
X_scaled_m_f2 = transformer.fit_transform(X[feature_set_2])
# Training the X_scaled on all the 4 models using both the feature_sets
print("\nFitting algorithms with feature_set_1 on MaxAbsScaled X data ...")
fit_algorithm("LR feature_set_1 MaxAbs", model_2, X_scaled_m_f1, y)
fit_algorithm("SGD feature_set_1 MaxAbs", model_3, X_scaled_m_f1, y)
fit_algorithm("GB feature_set_1 MaxAbs", model_4, X_scaled_m_f1, y)
print("\nFitting algorithms with feature_set_2 on MaxAbsScaled X data ...")
fit_algorithm("LR feature_set_2 MaxAbs", model_2, X_scaled_m_f2, y)
fit_algorithm("SGD feature_set_2 MaxAbs", model_3, X_scaled_m_f2, y)
fit_algorithm("GB feature_set_2 MaxAbs", model_4, X_scaled_m_f2, y)


###############################################################################

# Scaling the data using Standard Scaler to check if we can get more relaible models

scaler = StandardScaler()
X_scaled_s_f1 = scaler.fit_transform(X[feature_set_1])
X_scaled_s_f2 = scaler.fit_transform(X[feature_set_2])
# Training the X_scaled on all the 4 models using both the feature_sets
print("\nFitting algorithms with feature_set_1 on Standard Scaled X data ...")
fit_algorithm("LR feature_set_1 StdScale", model_2, X_scaled_s_f1, y)
fit_algorithm("SGD feature_set_1 StdScale", model_3, X_scaled_s_f1, y)
fit_algorithm("GB feature_set_1 StdScale", model_4, X_scaled_s_f1, y)
print("\nFitting algorithms with feature_set_2 on Standard Scaled X data ...")
fit_algorithm("LR feature_set_2 StdScale", model_2, X_scaled_s_f2, y)
fit_algorithm("SGD feature_set_2 StdScale", model_3, X_scaled_s_f2, y)
fit_algorithm("GB feature_set_2 StdScale", model_4, X_scaled_s_f2, y)


###############################################################################

# Feature sets taken from the Correlation analysis done in EDA script
feature_set_3 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_dob', 'nunique_language', 'nunique_report_type',
                 'nunique_device', 'transactions_amount']
feature_set_4 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_report_type', 'nunique_device', 'transactions_amount']

print(f"Feature set 3: {feature_set_3}")
print(f"Feature set 4: {feature_set_4}")

# Fitting algorithms on feature_set_3 and feature_set_4 using MaxAbsScaler
X_scaled_m_f3 = transformer.fit_transform(X[feature_set_3])
X_scaled_m_f4 = transformer.fit_transform(X[feature_set_4])
# Training the transformed feature sets on all models
# Fitting algorithms on X_scaled_m_f3
print("\nFitting algorithms with feature_set_3 on MaxAbsScaled X data ...")
fit_algorithm("RF feature_set_3 MaxAbs", model_1, X_scaled_m_f3, y)
fit_algorithm("LR feature_set_3 MaxAbs", model_2, X_scaled_m_f3, y)
fit_algorithm("SGD feature_set_3 MaxAbs", model_3, X_scaled_m_f3, y)
fit_algorithm("GB feature_set_3 MaxAbs", model_4, X_scaled_m_f3, y)
# Fitting algorithms on X_scaled_m_f4
print("\nFitting algorithms with feature_set_4 on MaxAbsScaled X data ...")
fit_algorithm("RF feature_set_4 MaxAbs", model_1, X_scaled_m_f4, y)
fit_algorithm("LR feature_set_4 MaxAbs", model_2, X_scaled_m_f4, y)
fit_algorithm("SGD feature_set_4 MaxAbs", model_3, X_scaled_m_f4, y)
fit_algorithm("GB feature_set_4 MaxAbs", model_4, X_scaled_m_f4, y)

# Fitting algorithms on feature_set_3 and feature_set_4 using MaxAbsScaler
scaler = StandardScaler()
X_scaled_s_f3 = scaler.fit_transform(X[feature_set_3])
X_scaled_s_f4 = scaler.fit_transform(X[feature_set_4])
# Training the transformed feature sets on all models
# Fitting algorithms on X_scaled_s_f3
print("\nFitting algorithms with feature_set_3 on Standard Scaled X data ...")
fit_algorithm("RF feature_set_3 StdScale", model_1, X_scaled_s_f3, y)
fit_algorithm("LR feature_set_3 StdScale", model_2, X_scaled_s_f3, y)
fit_algorithm("SGD feature_set_3 StdScale", model_3, X_scaled_s_f3, y)
fit_algorithm("GB feature_set_3 StdScale", model_4, X_scaled_s_f3, y)
# Fitting algorithms on X_scaled_s_f4
print("\nFitting algorithms with feature_set_4 on MaxAbsScaled X data ...")
fit_algorithm("RF feature_set_4 StdScale", model_1, X_scaled_s_f4, y)
fit_algorithm("LR feature_set_4 StdScale", model_2, X_scaled_s_f4, y)
fit_algorithm("SGD feature_set_4 StdScale", model_3, X_scaled_s_f4, y)

# Storing the final result table into a dataframe and converting it to a csv
model_evaluation_table = fit_algorithm("GB feature_set_4 StdScale", model_4, X_scaled_s_f4, y)
model_evaluation_table.to_csv('model_scores.csv', encoding='utf=-8')

# Training ends
print("\n{}\tTraining completed !!".format(datetime.datetime.now().strftime("'%Y-%m-%d %H:%M:%S'")))



###############################################################################


# 2021-09-22 09:16:03     Reading dataset: base_data_resampled_tomek.csv ...

# Splitting into features and lables ...
# Building evaluation and training functions ...

# Creating models RandomForests, LogisticRegression, StochasticGradientDescent and GradientBoostClassifier...
# Fitting algorithms taking all features ...
# Feature importances from Random forests are:-
#                       importance_magnitude
# features
# transactions_amount                 0.4185
# count_pay_attempt                   0.3139
# nunique_beacon_type                 0.0707
# count_user_stay                     0.0583
# count_buy_click                     0.0557
# profile_submit_count                0.0446
# sum_beacon_value                    0.0273
# count_sessions                      0.0035
# nunique_report_type                 0.0033
# nunique_dob                         0.0019
# nunique_gender                      0.0011
# nunique_device                      0.0006
# nunique_language                    0.0004
# Choosing the top 7 features to create feature_set_1 ...
# Feature set 1: ['transactions_amount', 'count_pay_attempt', 'nunique_beacon_type', 'count_user_stay', 'count_buy_click', 'profile_submit_count', 'sum_beacon_value']

# Fitting algorithms with feature_set_1 ...
# Feature importances from Logistic Regression are:-
#                       importance_magnitude
# features
# count_pay_attempt                   4.0124
# nunique_device                      0.9974
# nunique_report_type                 0.8022
# count_buy_click                     0.7840
# nunique_beacon_type                 0.3959
# count_sessions                      0.3513
# nunique_language                    0.3179
# nunique_dob                         0.1947
# nunique_gender                      0.0341
# profile_submit_count                0.0223
# count_user_stay                     0.0058
# transactions_amount                 0.0028
# sum_beacon_value                    0.0011
# Choosing the top 7 features from Logistic Regression model to create feature_set_2 ...
# Feature set 2: ['count_pay_attempt', 'nunique_device', 'nunique_report_type', 'count_buy_click', 'nunique_beacon_type', 'count_sessions', 'nunique_language']

# Fitting algorithms with feature_set_2 ...

# Fitting algorithms with feature_set_1 on MaxAbsScaled X data ...

# Fitting algorithms with feature_set_2 on MaxAbsScaled X data ...

# Fitting algorithms with feature_set_1 on Standard Scaled X data ...

# Fitting algorithms with feature_set_2 on Standard Scaled X data ...
# Feature set 3: ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click', 'nunique_dob', 'nunique_language', 'nunique_report_type', 'nunique_device', 'transactions_amount']
# Feature set 4: ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click', 'nunique_report_type', 'nunique_device', 'transactions_amount']

# Fitting algorithms with feature_set_3 on MaxAbsScaled X data ...

# Fitting algorithms with feature_set_4 on MaxAbsScaled X data ...

# Fitting algorithms with feature_set_3 on Standard Scaled X data ...

# Fitting algorithms with feature_set_4 on MaxAbsScaled X data ...

#                      model_name  feature_count  Balanced_Accuracy_test  Recall_test  Balanced_Accuracy_train  Recall_train  Fit_time  Score_time
# 19   GB feature_set_1 StdScale              7                   0.988        0.998                    0.989         0.999    12.757       0.063
# 6             GB feature_set_1              7                   0.988        0.998                    0.989         0.999    12.553       0.075
# 13     GB feature_set_1 MaxAbs              7                   0.988        0.998                    0.989         0.999    12.752       0.063
# 0              RF all features             13                   0.986        0.994                    0.996         0.999     9.111       0.368
# 3             RF feature_set_1              7                   0.986        0.994                    0.996         0.999     8.246       0.363
# 38   GB feature_set_4 StdScale              6                   0.986        0.999                    0.986         1.000     8.660       0.064
# 30     GB feature_set_4 MaxAbs              6                   0.986        0.999                    0.986         1.000     8.665       0.063
# 34   GB feature_set_3 StdScale              8                   0.986        0.999                    0.986         0.999     9.522       0.064
# 26     GB feature_set_3 MaxAbs              8                   0.986        0.999                    0.986         0.999     9.545       0.064
# 5            SGD feature_set_1              7                   0.984        0.995                    0.985         0.995     0.248       0.050
# 31   RF feature_set_3 StdScale              8                   0.984        0.994                    0.991         0.999     6.987       0.348
# 23     RF feature_set_3 MaxAbs              8                   0.984        0.994                    0.991         0.999     6.910       0.351
# 27     RF feature_set_4 MaxAbs              6                   0.984        0.994                    0.991         0.999     6.775       0.335
# 35   RF feature_set_4 StdScale              6                   0.984        0.994                    0.991         0.999     6.798       0.336
# 2             SGD all features             13                   0.982        0.988                    0.982         0.988     0.267       0.038
# 1              LR all features             13                   0.963        0.943                    0.962         0.942    26.461       0.037
# 4             LR feature_set_1              7                   0.962        0.942                    0.962         0.941     4.311       0.049
# 18  SGD feature_set_1 StdScale              7                   0.949        0.915                    0.950         0.916     0.340       0.039
# 33  SGD feature_set_3 StdScale              8                   0.949        0.914                    0.949         0.915     0.416       0.040
# 37  SGD feature_set_4 StdScale              6                   0.948        0.914                    0.948         0.914     0.330       0.038
# 36   LR feature_set_4 StdScale              6                   0.941        0.898                    0.941         0.898     0.476       0.037
# 17   LR feature_set_1 StdScale              7                   0.941        0.897                    0.941         0.897     0.576       0.037
# 32   LR feature_set_3 StdScale              8                   0.941        0.898                    0.941         0.898     0.535       0.038
# 24     LR feature_set_3 MaxAbs              8                   0.929        0.872                    0.929         0.872     1.180       0.038
# 28     LR feature_set_4 MaxAbs              6                   0.929        0.872                    0.929         0.872     0.983       0.039
# 11     LR feature_set_1 MaxAbs              7                   0.929        0.871                    0.929         0.871     1.098       0.037
# 10            GB feature_set_1              7                   0.928        0.872                    0.930         0.873     6.841       0.073
# 22   GB feature_set_2 StdScale              7                   0.928        0.872                    0.930         0.873     7.028       0.063
# 16     GB feature_set_2 MaxAbs              7                   0.928        0.872                    0.930         0.873     7.006       0.063
# 8             LR feature_set_1              7                   0.928        0.871                    0.929         0.871     1.172       0.048
# 20   LR feature_set_2 StdScale              7                   0.928        0.871                    0.929         0.871     0.374       0.038
# 14     LR feature_set_2 MaxAbs              7                   0.928        0.871                    0.928         0.871     1.256       0.038
# 25    SGD feature_set_3 MaxAbs              8                   0.928        0.869                    0.928         0.869     0.235       0.039
# 12    SGD feature_set_1 MaxAbs              7                   0.928        0.869                    0.928         0.869     0.203       0.038
# 29    SGD feature_set_4 MaxAbs              6                   0.928        0.869                    0.928         0.869     0.205       0.038
# 21  SGD feature_set_2 StdScale              7                   0.928        0.869                    0.928         0.868     0.405       0.039
# 9            SGD feature_set_1              7                   0.928        0.868                    0.928         0.868     0.531       0.050
# 15    SGD feature_set_2 MaxAbs              7                   0.928        0.868                    0.928         0.868     0.238       0.038
# 7             RF feature_set_1              7                   0.928        0.872                    0.931         0.875     5.225       0.326

# '2021-09-22 09:17:19'   Training completed !!