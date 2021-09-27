# This script cross validates the models on the tuned hyperparameters obtained
# from the Hyperparameter_Tuning script and choosed the best one

import pandas as pd
import os
import datetime
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier as SDC
from sklearn.preprocessing import StandardScaler

# Reading the dataset base_data_resampled_tomek.csv
base_path = os.path.dirname(os.path.realpath(__file__))
in_file_name = "base_data_resampled_tomek.csv"
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name))
df_test = pd.read_csv(os.path.join(base_path, in_file_name))

# Seperating the features and labels
print("\nSplitting into features and lables ...")
X = df_test.drop(columns=['conversion_status', 'email', 'date'], axis=1)
y = df_test['conversion_status']

# Reading the tuned parameters corresponding to each model
base_path = os.path.dirname(os.path.realpath(__file__))
in_file_name1 = "Logistic_Tuned_Hyperparameters.csv"
in_file_name2 = "SGD_Tuned_Hyperparameters.csv"
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name1))
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name2))
df_logit_params = pd.read_csv(os.path.join(base_path, in_file_name1))
df_sgd_params = pd.read_csv(os.path.join(base_path, in_file_name2))

print("The best parameters for LOGISTIC REGRESSION models are: -")
print(df_logit_params.to_string())
print("\nThe best parameters for STOCHASTIC GRADIENT DESCENT models are: -")
print(df_sgd_params.to_string())


# Listing the feature sets selected from the training script
print("\nCreating feature sets ...")
feature_set_1 = ['transactions_amount', 'count_pay_attempt', 'nunique_beacon_type',
                 'count_user_stay', 'count_buy_click', 'profile_submit_count',
                 'sum_beacon_value']
feature_set_3 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_dob', 'nunique_language', 'nunique_report_type',
                 'nunique_device', 'transactions_amount']
feature_set_4 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                 'nunique_report_type', 'nunique_device', 'transactions_amount']

feature_set_5 = ['count_pay_attempt', 'count_buy_click',
                 'nunique_report_type', 'profile_submit_count']


print("\nScaling features ...")
scaler = StandardScaler()
X_scaled_s_f1 = scaler.fit_transform(X[feature_set_1])
X_scaled_s_f3 = scaler.fit_transform(X[feature_set_3])
X_scaled_s_f4 = scaler.fit_transform(X[feature_set_4])
X_scaled_s_f5 = scaler.fit_transform(X[feature_set_5])


print("Building evaluation and training functions ...")

##############################################################################

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

def model_evaluation(model_name, fc, bac_test, rcc_test, bac_train, rcc_train, ft, st):
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

# Since, we need to partial fit later, which can be achieved with the help of SGD
# Best hyperparameters for STOCHASTIC GRADIENT DESCENT: -
#                          model                                           best hyperparameters
# 0  SGD feature_set_1 StdScaled  {'alpha': 1e-05, 'class_weight': 'balanced', 'penalty': 'l1'}
# 1  SGD feature_set_3 StdScaled  {'alpha': 1e-05, 'class_weight': 'balanced', 'penalty': 'l1'}
# 2  SGD feature_set_4 StdScaled  {'alpha': 1e-05, 'class_weight': 'balanced', 'penalty': 'l1'}
# 3  SGD feature_set_5 StdScaled   {'alpha': 0.01, 'class_weight': 'balanced', 'penalty': 'l2'}

###############################################################################

print("SGD begins ...")
sgd_model_tuned1 = SDC(random_state=23, max_iter=3000, loss='log', alpha=0.00001, class_weight='balanced', penalty='l1')
sgd_model_tuned2 = SDC(random_state=23, max_iter=3000, loss='log', alpha=0.00001, penalty='elasticnet')
sgd_model_tuned3 = SDC(random_state=23, max_iter=3000, loss='log', alpha=0.01, class_weight='balanced', penalty='l2')

fit_algorithm("SDC1 feature_set_1 std scaled tuned", sgd_model_tuned1, X_scaled_s_f1, y)
fit_algorithm("SDC1 feature_set_3 std scaled tuned", sgd_model_tuned1, X_scaled_s_f3, y)
fit_algorithm("SDC1 feature_set_4 std scaled tuned", sgd_model_tuned1, X_scaled_s_f4, y)
fit_algorithm("SDC1 feature_set_5 std scaled tuned", sgd_model_tuned1, X_scaled_s_f5, y)

fit_algorithm("SDC2 feature_set_1 std scaled tuned", sgd_model_tuned2, X_scaled_s_f1, y)
fit_algorithm("SDC2 feature_set_3 std scaled tuned", sgd_model_tuned2, X_scaled_s_f3, y)
fit_algorithm("SDC2 feature_set_4 std scaled tuned", sgd_model_tuned2, X_scaled_s_f4, y)
fit_algorithm("SDC2 feature_set_5 std scaled tuned", sgd_model_tuned2, X_scaled_s_f5, y)

fit_algorithm("SDC3 feature_set_1 std scaled tuned", sgd_model_tuned3, X_scaled_s_f1, y)
fit_algorithm("SDC3 feature_set_3 std scaled tuned", sgd_model_tuned3, X_scaled_s_f3, y)
fit_algorithm("SDC3 feature_set_4 std scaled tuned", sgd_model_tuned3, X_scaled_s_f4, y)
df_final_tuned = fit_algorithm("SDC3 feature_set_5 std scaled tuned", sgd_model_tuned3, X_scaled_s_f5, y)

print(df_final_tuned.to_string())
print("Creating csv ...")
df_final_tuned.to_csv('LR SGD tuned_models.csv', encoding='utf-8', index=False)
print("Done")

###############################################################################


# 2021-09-27 08:37:26     Reading dataset: base_data_resampled_tomek.csv ...

# Splitting into features and lables ...

# 2021-09-27 08:37:26     Reading dataset: Logistic_Tuned_Hyperparameters.csv ...

# 2021-09-27 08:37:26     Reading dataset: SGD_Tuned_Hyperparameters.csv ...
# The best parameters for LOGISTIC REGRESSION models are: -
#                          model                                           best hyperparameters
# 0  LR1 feature_set_1 StdScaled  {'C': 0.1, 'class_weight': 'balanced', 'solver': 'newton-cg'}
# 1  LR1 feature_set_4 StdScaled  {'C': 0.1, 'class_weight': 'balanced', 'solver': 'newton-cg'}
# 2  LR2 feature_set_1 StdScaled          {'C': 1, 'class_weight': 'balanced', 'penalty': 'l1'}
# 3  LR2 feature_set_4 StdScaled          {'C': 1, 'class_weight': 'balanced', 'penalty': 'l1'}

# The best parameters for STOCHASTIC GRADIENT DESCENT models are: -
#                          model                                           best hyperparameters
# 0  SGD feature_set_1 StdScaled  {'alpha': 1e-05, 'class_weight': 'balanced', 'penalty': 'l1'}
# 1  SGD feature_set_3 StdScaled  {'alpha': 1e-05, 'class_weight': 'balanced', 'penalty': 'l1'}
# 2  SGD feature_set_4 StdScaled  {'alpha': 1e-05, 'class_weight': 'balanced', 'penalty': 'l1'}
# 3  SGD feature_set_5 StdScaled   {'alpha': 0.01, 'class_weight': 'balanced', 'penalty': 'l2'}
# Creating feature sets ...
# Scaling features ...
# Building evaluation and training functions ...
# SGD begins ...
#                              model_name  feature_count  Balanced_Accuracy_test  Recall_test  Balanced_Accuracy_train  Recall_train  Fit_time  Score_time
# 0   SDC1 feature_set_1 std scaled tuned              7                   0.979        0.980                    0.979         0.980     1.813       0.041
# 1   SDC1 feature_set_3 std scaled tuned              8                   0.976        0.978                    0.976         0.978     1.468       0.039
# 2   SDC1 feature_set_4 std scaled tuned              6                   0.967        0.955                    0.968         0.957     1.285       0.046
# 5   SDC2 feature_set_3 std scaled tuned              8                   0.956        0.930                    0.955         0.929     1.244       0.040
# 6   SDC2 feature_set_4 std scaled tuned              6                   0.956        0.930                    0.955         0.928     1.264       0.046
# 4   SDC2 feature_set_1 std scaled tuned              7                   0.955        0.927                    0.956         0.927     1.504       0.046
# 10  SDC3 feature_set_4 std scaled tuned              6                   0.929        0.874                    0.929         0.874     0.283       0.040
# 11  SDC3 feature_set_5 std scaled tuned              4                   0.929        0.874                    0.929         0.874     0.252       0.039
# 9   SDC3 feature_set_3 std scaled tuned              8                   0.929        0.873                    0.929         0.873     0.306       0.039
# 3   SDC1 feature_set_5 std scaled tuned              4                   0.929        0.872                    0.928         0.871     1.113       0.054
# 8   SDC3 feature_set_1 std scaled tuned              7                   0.928        0.872                    0.928         0.872     0.295       0.040
# 7   SDC2 feature_set_5 std scaled tuned              4                   0.928        0.871                    0.928         0.872     1.029       0.039
# Creating csv ...
# Done



################################################################################