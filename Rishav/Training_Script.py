import pandas as pd
import os
import datetime
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LRC, SGDClassifier as SDC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

base_path = os.path.dirname(os.path.realpath(__file__))
in_file_name = "resampled_data_tomek.csv"
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name))
print("\nSplitting into features and lables ...")
df_test = pd.read_csv(os.path.join(base_path, in_file_name))

# Dividing the features and labels
X = df_test.drop('conversion_status', axis=1)
y = df_test['conversion_status']


print("\n\t{}Building evaluation and training functions ...".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
##########################################

evaluation = {'model': [],
              'feature_count': [],
              'BAC': [],
              'recall': []
              } 

def model_evaluation(model_name, fc, bac, rcc):
    """
    To keep track of the models used and their
    respective characterisitics with evaluation scores.
    """
    evaluation['model'].append(model_name)
    evaluation['feature_count'].append(fc)
    evaluation['BAC'].append(bac)
    evaluation['recall'].append(rcc)
    
    df_eval = pd.DataFrame({'model_name': evaluation['model'],
                            'feature_count': evaluation['feature_count'],
                            'BAC': evaluation['BAC'],
                            'REC': evaluation['recall'] 
                            })
    return df_eval.sort_values(by='BAC', ascending=False).round(3)


def fit_algorithm(algo_name, algorithm, X, y, cv=5):
    """
    Accepts data from the prepare data function and 
    return balanced accuracy scores of the predicted model.
    """
    model = cross_validate(algorithm, X, y, cv=cv, n_jobs=-1, scoring=['balanced_accuracy', 'recall'])
    df = model_evaluation(algo_name, X.shape[1], model['test_balanced_accuracy'].mean(), model['test_recall'].mean())
    
    return df


######################################################
print("\nFitting algorithms with all features ...")
fit_algorithm("RF all features", RFC(random_state=23), X.to_numpy(), y, cv=5)
fit_algorithm("LR all features", LRC(random_state=23, max_iter=5000), X.to_numpy(), y, cv=5)
fit_algorithm("SGD all features", SDC(random_state=23), X.to_numpy(), y, cv=5)

# Fitting algorithms with features derived from EDA

print("\nCreating 4 feature sets and training ...")
feature_set_1 = ['count_sessions',                       # Feature set 1 with least correlated value with y dropped
                 'sum_beacon_value',
                 'nunique_beacon_type',
                 'count_pay_attempt',
                 'count_buy_click',
                 'nunique_gender',
                 'nunique_dob',
                 'nunique_language',
                 'nunique_report_type',
                 'nunique_device',
                 'profile_submit_count',
                 'transactions_amount']

# Fitting 3 algorithms using feature set 1
fit_algorithm("RF feature_set_1", RFC(random_state=23), X[feature_set_1], y, cv=5)
fit_algorithm("LR feature_set_1", LRC(random_state=23, max_iter=5000), X[feature_set_1], y, cv=5)
fit_algorithm("SGD feature_set_1", SDC(random_state=23), X[feature_set_1], y, cv=5)


feature_set_2 = ['count_sessions',                       
                 'sum_beacon_value',
                 'nunique_beacon_type',
                 'count_pay_attempt',
                 'count_buy_click',
                 'nunique_gender',
                 'nunique_dob',
                 'nunique_report_type',
                 'nunique_device',
                 'profile_submit_count',
                 'transactions_amount']

# Fitting 3 algorithms using feature set 2
fit_algorithm("RF feature_set_2", RFC(random_state=23), X[feature_set_2], y, cv=5)
fit_algorithm("LR feature_set_2", LRC(random_state=23, max_iter=5000), X[feature_set_2], y, cv=5)
fit_algorithm("SGD feature_set_2", SDC(random_state=23), X[feature_set_2], y, cv=5)


feature_set_3 = ['sum_beacon_value',
                 'nunique_beacon_type',
                 'count_pay_attempt',
                 'count_buy_click',
                 'nunique_gender',
                 'nunique_dob',
                 'nunique_language', 
                 'nunique_report_type',
                 'nunique_device',
                 'profile_submit_count',
                 'transactions_amount']
# Fitting 3 algorithms using feautre set 3
fit_algorithm("RF feature_set_3", RFC(random_state=23), X[feature_set_3], y, cv=5)
fit_algorithm("LR feature_set_3", LRC(random_state=23, max_iter=5000), X[feature_set_3], y, cv=5)
fit_algorithm("SGD feature_set_3", SDC(random_state=23), X[feature_set_3], y, cv=5)

feature_set_4 = ['sum_beacon_value',
                 'count_pay_attempt',
                 'count_buy_click',
                 'nunique_gender',
                 'nunique_dob',
                 'nunique_language',
                 'nunique_report_type',
                 'nunique_device',
                 'profile_submit_count',
                 'transactions_amount']
# Fitting 3 algorithms with feature set 4
fit_algorithm("RF feature_set_4", RFC(random_state=23), X[feature_set_4], y, cv=5)
fit_algorithm("LR feature_set_4", LRC(random_state=23, max_iter=5000), X[feature_set_4], y, cv=5)
fit_algorithm("SGD feature_set_4", SDC(random_state=23), X[feature_set_4], y, cv=5)

##########################################################

print("\nMax Absolute Scaling the data ...")
transformer = MaxAbsScaler()                                       # Object to scale the data using max absolute scaler
# Scaling the independent variables using max abs scaler
X_feature_set_1_mscaled = transformer.fit_transform(X[feature_set_1])
X_feature_set_2_mscaled = transformer.fit_transform(X[feature_set_2])
X_feature_set_3_mscaled = transformer.fit_transform(X[feature_set_3])
X_feature_set_4_mscaled = transformer.fit_transform(X[feature_set_4])
print("\nTraining on the scaled data ...")

# Fitting 4 models using the scaled data with feature set 1
fit_algorithm("RF feature_set_1 maxabs scaled", RFC(random_state=23), X_feature_set_1_mscaled, y, cv=5)
fit_algorithm("LR feature_set_1 maxabs scaled", LRC(random_state=23, max_iter=1500), X_feature_set_1_mscaled, y, cv=5)
fit_algorithm("SGD feature_set_1 maxabs scaled", SDC(random_state=23), X_feature_set_1_mscaled, y, cv=5)
fit_algorithm("GBoost feature_set_1 maxabs scaled",GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_1_mscaled, y, cv=5)

# Fitting 4 models using the scaled data with feature set 2
fit_algorithm("RF feature_set_2 maxabs scaled", RFC(random_state=23), X_feature_set_2_mscaled, y, cv=5)
fit_algorithm("LR feature_set_2 maxabs scaled", LRC(random_state=23, max_iter=1500), X_feature_set_2_mscaled, y, cv=5)
fit_algorithm("SGD feature_set_2 maxabs scaled", SDC(random_state=23), X_feature_set_2_mscaled, y, cv=5)
fit_algorithm("GBoost feature_set_2 maxabs scaled",GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_2_mscaled, y, cv=5)


# Fitting 4 models using the scaled data with feature set 3
fit_algorithm("RF feature_set_3 maxabs scaled", RFC(random_state=23), X_feature_set_3_mscaled, y, cv=5)
fit_algorithm("LR feature_set_3 maxabs scaled", LRC(random_state=23, max_iter=1500), X_feature_set_3_mscaled, y, cv=5)
fit_algorithm("SGD feature_set_3 maxabs scaled", SDC(random_state=23), X_feature_set_3_mscaled, y, cv=5)
fit_algorithm("GBoost feature_set_3 maxabs scaled",GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_3_mscaled, y, cv=5)

# Fitting 4 models using the scaled data with feature set 4
fit_algorithm("RF feature_set_4 maxabs scaled", RFC(random_state=23), X_feature_set_4_mscaled, y, cv=5)
fit_algorithm("LR feature_set_4 maxabs scaled", LRC(random_state=23, max_iter=1500), X_feature_set_4_mscaled, y, cv=5)
fit_algorithm("SGD feature_set_4 maxabs scaled", SDC(random_state=23), X_feature_set_4_mscaled, y, cv=5)
fit_algorithm("GBoost feature_set_4 maxabs scaled",GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_4_mscaled, y, cv=5)

print("\nStandard scaling the data ...")
scaler = StandardScaler()                                         # Object to scale the data using standard scaler
# Standard scaling the data
X_feature_set_1_sscaled = scaler.fit_transform(X[feature_set_1])
X_feature_set_2_sscaled = scaler.fit_transform(X[feature_set_2])
X_feature_set_3_sscaled = scaler.fit_transform(X[feature_set_3])
X_feature_set_4_sscaled = scaler.fit_transform(X[feature_set_4])

# Fitting 4 models using the standard scaled data with feature set 1
print("\nTraining on the standardized data ...")
fit_algorithm("RF feature_set_1 standard scaled", RFC(random_state=23), X_feature_set_1_sscaled, y, cv=5)
fit_algorithm("LR feature_set_1 standard scaled", LRC(random_state=23, max_iter=1000), X_feature_set_1_sscaled, y, cv=5)
fit_algorithm("SGD feature_set_1 standard scaled", SDC(random_state=23), X_feature_set_1_sscaled, y, cv=5)
fit_algorithm("GBoost feature_set_1 standard scaled", GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_1_sscaled, y, cv=5)

# Fitting 4 models using the standard scaled data with feature set 2
fit_algorithm("RF feature_set_2 standard scaled", RFC(random_state=23), X_feature_set_2_sscaled, y, cv=5)
fit_algorithm("LR feature_set_2 standard scaled", LRC(random_state=23, max_iter=1000), X_feature_set_2_sscaled, y, cv=5)
fit_algorithm("SGD feature_set_2 standard scaled", SDC(random_state=23), X_feature_set_2_sscaled, y, cv=5)
fit_algorithm("XGBoost feature_set_2 standard scaled", GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_2_sscaled, y, cv=5)

# Fitting 4 models using the standard scaled data with feature set 3
fit_algorithm("RF feature_set_3 standard scaled", RFC(random_state=23), X_feature_set_3_sscaled, y, cv=5)
fit_algorithm("LR feature_set_3 standard scaled", LRC(random_state=23, max_iter=1000), X_feature_set_3_sscaled, y, cv=5)
fit_algorithm("SGD feature_set_3 standard scaled", SDC(random_state=23), X_feature_set_3_sscaled, y, cv=5)
fit_algorithm("GBoost feature_set_3 standard scaled", GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_3_sscaled, y, cv=5)

# Fitting 4 models using the standard scaled data with feature set 4
fit_algorithm("RF feature_set_4 standard scaled", RFC(random_state=23), X_feature_set_4_sscaled, y, cv=5)
fit_algorithm("LR feature_set_4 standard scaled", LRC(random_state=23, max_iter=1000), X_feature_set_4_sscaled, y, cv=5)
fit_algorithm("SGD feature_set_4 standard scaled", SDC(random_state=23), X_feature_set_4_sscaled, y, cv=5)
print(fit_algorithm("GBoost feature_set_2 standard scaled", GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=23), X_feature_set_4_sscaled, y, cv=5))
print("\n{}\tTraining completed !!".format(datetime.datetime.now().strftime("'%Y-%m-%d %H:%M:%S'")))

##################################################################

# 2021-09-21 08:33:10     Reading dataset: resampled_data_tomek.csv ...

# Splitting into features and lables ...

# 2021-09-21 08:33:10Building evaluation and training functions ...

# Fitting algorithms with all features ...

# Creating 4 feature sets and training ...

# Max Absolute Scaling the data ...

# Training on the scaled data ...

# Standard scaling the data ...

# Training on the standardized data ...
#                                model_name  feature_count    BAC    REC
# 46   GBoost feature_set_2 standard scaled             10  0.981  0.997
# 30     GBoost feature_set_4 maxabs scaled             10  0.981  0.997
# 38  GBoost feature_set_2 standard scaled             11  0.981  0.996
# 34   GBoost feature_set_1 standard scaled             12  0.981  0.996
# 22     GBoost feature_set_2 maxabs scaled             11  0.981  0.996
# 18     GBoost feature_set_1 maxabs scaled             12  0.981  0.996
# 42   GBoost feature_set_3 standard scaled             11  0.981  0.997
# 26     GBoost feature_set_3 maxabs scaled             11  0.981  0.997
# 15         RF feature_set_1 maxabs scaled             12  0.979  0.991
# 3                        RF feature_set_1             12  0.979  0.991
# 35       RF feature_set_2 standard scaled             11  0.979  0.991
# 31       RF feature_set_1 standard scaled             12  0.979  0.991
# 19         RF feature_set_2 maxabs scaled             11  0.979  0.991
# 6                        RF feature_set_2             11  0.979  0.991
# 39       RF feature_set_3 standard scaled             11  0.979  0.991
# 23         RF feature_set_3 maxabs scaled             11  0.979  0.991
# 9                        RF feature_set_3             11  0.979  0.991
# 27         RF feature_set_4 maxabs scaled             10  0.979  0.990
# 12                       RF feature_set_4             10  0.979  0.990
# 43       RF feature_set_4 standard scaled             10  0.979  0.990
# 0                         RF all features             13  0.978  0.990
# 14                      SGD feature_set_4             10  0.977  0.992
# 2                        SGD all features             13  0.977  0.991
# 8                       SGD feature_set_2             11  0.977  0.991
# 5                       SGD feature_set_1             12  0.976  0.991
# 11                      SGD feature_set_3             11  0.975  0.987
# 7                        LR feature_set_2             11  0.943  0.910
# 10                       LR feature_set_3             11  0.943  0.909
# 1                         LR all features             13  0.943  0.909
# 13                       LR feature_set_4             10  0.943  0.909
# 4                        LR feature_set_1             12  0.943  0.909
# 45      SGD feature_set_4 standard scaled             10  0.938  0.897
# 33      SGD feature_set_1 standard scaled             12  0.937  0.897
# 37      SGD feature_set_2 standard scaled             11  0.937  0.895
# 41      SGD feature_set_3 standard scaled             11  0.936  0.894
# 32       LR feature_set_1 standard scaled             12  0.932  0.884
# 36       LR feature_set_2 standard scaled             11  0.932  0.884
# 44       LR feature_set_4 standard scaled             10  0.932  0.884
# 40       LR feature_set_3 standard scaled             11  0.932  0.884
# 28         LR feature_set_4 maxabs scaled             10  0.926  0.871
# 24         LR feature_set_3 maxabs scaled             11  0.925  0.871
# 16         LR feature_set_1 maxabs scaled             12  0.925  0.871
# 20         LR feature_set_2 maxabs scaled             11  0.925  0.871
# 17        SGD feature_set_1 maxabs scaled             12  0.925  0.869
# 29        SGD feature_set_4 maxabs scaled             10  0.925  0.869
# 25        SGD feature_set_3 maxabs scaled             11  0.925  0.869
# 21        SGD feature_set_2 maxabs scaled             11  0.925  0.869

# Training completed !! '2021-09-21 08:33:59'