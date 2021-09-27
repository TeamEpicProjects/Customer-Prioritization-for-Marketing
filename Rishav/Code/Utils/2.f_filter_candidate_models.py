# This script filters models based on the bac and recall scores
# given by the domain expert

import pandas as pd
import os


base_path = os.path.dirname(os.path.realpath(__file__))

def df_info(df):
    col_name_list = list(df.columns)
    col_type_list = [type(col) for col in df.iloc[0, :]]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_memory_usage_list = [df[col].memory_usage(deep=True) for col in col_name_list]
    df_total_memory_usage = sum(col_memory_usage_list) / 1048576
    return pd.DataFrame({'col_name': col_name_list, 'col_type': col_type_list, 'null_count': col_null_count_list, 'nunique': col_unique_count_list}), df_total_memory_usage

##############################################################################

df_models = pd.read_csv('model_scores.csv')
df_models.columns = ['model_name', 'feature_count', 'bac_test', 'rec_test',
                     'bac_train', 'rec_train', 'time_fit', 'time_score']

df_models_info = df_info(df_models)
print('\n"models" dataset summary:')
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_models.shape[0], df_models.shape[1], df_models_info[1]))
print(df_models_info[0].to_string())
print('\n"models" dataset head:')
print(df_models.to_string())

df_filtered_models = df_models.loc[(df_models.bac_train < 0.96) & (df_models.rec_train < 0.96) & (df_models.bac_test >= 0.92) & (df_models.rec_test >= 0.86)]

print('\n"filtered_models" dataset summary:')
print('\t{} rows x {} columns'.format(df_filtered_models.shape[0], df_filtered_models.shape[1]))
print(df_filtered_models.to_string())


print("We can partial fit only using Stochastic Gradient Descent. Choosing SGD models...")
df_filtered_models_SGD = df_filtered_models.loc[df_filtered_models['model_name'].str.contains('SGD')]
print('\n"filtered_models_SGD" dataset summary:')
print('\t{} rows x {} columns'.format(df_filtered_models_SGD.shape[0], df_filtered_models_SGD.shape[1]))
print(df_filtered_models_SGD.to_string())



###############################################################################


# "models" dataset summary:
#         47 rows x 8 columns | 0.01 MB approx memory usage
#         col_name                 col_type  null_count  nunique
# 0     model_name            <class 'str'>           0       43
# 1  feature_count    <class 'numpy.int64'>           0        5
# 2       bac_test  <class 'numpy.float64'>           0       12
# 3       rec_test  <class 'numpy.float64'>           0       18
# 4      bac_train  <class 'numpy.float64'>           0       17
# 5      rec_train  <class 'numpy.float64'>           0       19
# 6       time_fit  <class 'numpy.float64'>           0       47
# 7     time_score  <class 'numpy.float64'>           0       23

# "models" dataset head:
#                     model_name  feature_count  bac_test  rec_test  bac_train  rec_train  time_fit  time_score
# 0             GB feature_set_1              7     0.988     0.998      0.989      0.999    11.573       0.071
# 1    GB feature_set_1 StdScale              7     0.988     0.998      0.989      0.999    11.753       0.064
# 2      GB feature_set_1 MaxAbs              7     0.988     0.998      0.989      0.999    12.348       0.064
# 3              RF all features             13     0.986     0.994      0.996      0.999     8.900       0.364
# 4             RF feature_set_1              7     0.986     0.994      0.996      0.999     8.276       0.354
# 5    GB feature_set_4 StdScale              6     0.986     0.999      0.986      1.000     8.204       0.060
# 6      GB feature_set_4 MaxAbs              6     0.986     0.999      0.986      1.000     8.378       0.062
# 7    GB feature_set_3 StdScale              8     0.986     0.999      0.986      0.999     9.226       0.061
# 8      GB feature_set_3 MaxAbs              8     0.986     0.999      0.986      0.999     9.086       0.060
# 9            SGD feature_set_1              7     0.984     0.995      0.985      0.995     0.242       0.049
# 10   RF feature_set_3 StdScale              8     0.984     0.994      0.991      0.999     6.579       0.317
# 11     RF feature_set_3 MaxAbs              8     0.984     0.994      0.991      0.999     6.593       0.323
# 12     RF feature_set_4 MaxAbs              6     0.984     0.994      0.991      0.999     6.660       0.316
# 13   RF feature_set_4 StdScale              6     0.984     0.994      0.991      0.999     6.506       0.316
# 14            SGD all features             13     0.982     0.988      0.982      0.988     0.277       0.039
# 15             LR all features             13     0.963     0.943      0.962      0.942    24.904       0.035
# 16            LR feature_set_1              7     0.962     0.942      0.962      0.941     4.234       0.048
# 17  SGD feature_set_1 StdScale              7     0.949     0.915      0.950      0.916     0.330       0.038
# 18  SGD feature_set_3 StdScale              8     0.949     0.914      0.949      0.915     0.412       0.038
# 19  SGD feature_set_4 StdScale              6     0.948     0.914      0.948      0.914     0.329       0.037
# 20   LR feature_set_4 StdScale              6     0.941     0.898      0.941      0.898     0.448       0.035
# 21   LR feature_set_1 StdScale              7     0.941     0.897      0.941      0.897     0.553       0.036
# 22   LR feature_set_3 StdScale              8     0.941     0.898      0.941      0.898     0.510       0.035
# 23   GB feature_set_5 StdScale              4     0.930     0.888      0.932      0.890     6.352       0.063
# 24     GB feature_set_5 MaxAbs              4     0.930     0.888      0.932      0.890     6.227       0.062
# 25     RF feature_set_5 MaxAbs              4     0.930     0.891      0.936      0.896     6.107       0.336
# 26   RF feature_set_5 StdScale              4     0.930     0.890      0.936      0.896     6.002       0.333
# 27     LR feature_set_3 MaxAbs              8     0.929     0.872      0.929      0.872     1.115       0.035
# 28     LR feature_set_4 MaxAbs              6     0.929     0.872      0.929      0.872     0.907       0.035
# 29     LR feature_set_5 MaxAbs              4     0.929     0.871      0.929      0.871     0.518       0.035
# 30     LR feature_set_1 MaxAbs              7     0.929     0.871      0.929      0.871     1.046       0.036
# 31   LR feature_set_5 StdScale              4     0.928     0.871      0.929      0.871     0.270       0.035
# 32     GB feature_set_2 MaxAbs              7     0.928     0.872      0.930      0.873     6.722       0.060
# 33   GB feature_set_2 StdScale              7     0.928     0.872      0.930      0.873     6.829       0.061
# 34            GB feature_set_1              7     0.928     0.872      0.930      0.873     6.598       0.071
# 35            LR feature_set_1              7     0.928     0.871      0.929      0.871     1.141       0.046
# 36   LR feature_set_2 StdScale              7     0.928     0.871      0.929      0.871     0.347       0.035
# 37     LR feature_set_2 MaxAbs              7     0.928     0.871      0.928      0.871     1.226       0.037
# 38    SGD feature_set_1 MaxAbs              7     0.928     0.869      0.928      0.869     0.200       0.039
# 39    SGD feature_set_3 MaxAbs              8     0.928     0.869      0.928      0.869     0.226       0.040
# 40    SGD feature_set_4 MaxAbs              6     0.928     0.869      0.928      0.869     0.201       0.037
# 41  SGD feature_set_2 StdScale              7     0.928     0.869      0.928      0.868     0.394       0.037
# 42    SGD feature_set_5 MaxAbs              4     0.928     0.868      0.928      0.868     0.173       0.038
# 43           SGD feature_set_1              7     0.928     0.868      0.928      0.868     0.502       0.048
# 44    SGD feature_set_2 MaxAbs              7     0.928     0.868      0.928      0.868     0.223       0.037
# 45  SGD feature_set_5 StdScale              4     0.928     0.868      0.928      0.868     0.291       0.037
# 46            RF feature_set_1              7     0.928     0.872      0.931      0.875     4.964       0.301

# "filtered_models" dataset summary:
#         30 rows x 8 columns
#                     model_name  feature_count  bac_test  rec_test  bac_train  rec_train  time_fit  time_score
# 17  SGD feature_set_1 StdScale              7     0.949     0.915      0.950      0.916     0.330       0.038
# 18  SGD feature_set_3 StdScale              8     0.949     0.914      0.949      0.915     0.412       0.038
# 19  SGD feature_set_4 StdScale              6     0.948     0.914      0.948      0.914     0.329       0.037
# 20   LR feature_set_4 StdScale              6     0.941     0.898      0.941      0.898     0.448       0.035
# 21   LR feature_set_1 StdScale              7     0.941     0.897      0.941      0.897     0.553       0.036
# 22   LR feature_set_3 StdScale              8     0.941     0.898      0.941      0.898     0.510       0.035
# 23   GB feature_set_5 StdScale              4     0.930     0.888      0.932      0.890     6.352       0.063
# 24     GB feature_set_5 MaxAbs              4     0.930     0.888      0.932      0.890     6.227       0.062
# 25     RF feature_set_5 MaxAbs              4     0.930     0.891      0.936      0.896     6.107       0.336
# 26   RF feature_set_5 StdScale              4     0.930     0.890      0.936      0.896     6.002       0.333
# 27     LR feature_set_3 MaxAbs              8     0.929     0.872      0.929      0.872     1.115       0.035
# 28     LR feature_set_4 MaxAbs              6     0.929     0.872      0.929      0.872     0.907       0.035
# 29     LR feature_set_5 MaxAbs              4     0.929     0.871      0.929      0.871     0.518       0.035
# 30     LR feature_set_1 MaxAbs              7     0.929     0.871      0.929      0.871     1.046       0.036
# 31   LR feature_set_5 StdScale              4     0.928     0.871      0.929      0.871     0.270       0.035
# 32     GB feature_set_2 MaxAbs              7     0.928     0.872      0.930      0.873     6.722       0.060
# 33   GB feature_set_2 StdScale              7     0.928     0.872      0.930      0.873     6.829       0.061
# 34            GB feature_set_1              7     0.928     0.872      0.930      0.873     6.598       0.071
# 35            LR feature_set_1              7     0.928     0.871      0.929      0.871     1.141       0.046
# 36   LR feature_set_2 StdScale              7     0.928     0.871      0.929      0.871     0.347       0.035
# 37     LR feature_set_2 MaxAbs              7     0.928     0.871      0.928      0.871     1.226       0.037
# 38    SGD feature_set_1 MaxAbs              7     0.928     0.869      0.928      0.869     0.200       0.039
# 39    SGD feature_set_3 MaxAbs              8     0.928     0.869      0.928      0.869     0.226       0.040
# 40    SGD feature_set_4 MaxAbs              6     0.928     0.869      0.928      0.869     0.201       0.037
# 41  SGD feature_set_2 StdScale              7     0.928     0.869      0.928      0.868     0.394       0.037
# 42    SGD feature_set_5 MaxAbs              4     0.928     0.868      0.928      0.868     0.173       0.038
# 43           SGD feature_set_1              7     0.928     0.868      0.928      0.868     0.502       0.048
# 44    SGD feature_set_2 MaxAbs              7     0.928     0.868      0.928      0.868     0.223       0.037
# 45  SGD feature_set_5 StdScale              4     0.928     0.868      0.928      0.868     0.291       0.037
# 46            RF feature_set_1              7     0.928     0.872      0.931      0.875     4.964       0.301
# We can partial fit only using Stochastic Gradient Descent. Choosing SGD models...

# "filtered_models_SGD" dataset summary:
#         11 rows x 8 columns
#                     model_name  feature_count  bac_test  rec_test  bac_train  rec_train  time_fit  time_score
# 17  SGD feature_set_1 StdScale              7     0.949     0.915      0.950      0.916     0.330       0.038
# 18  SGD feature_set_3 StdScale              8     0.949     0.914      0.949      0.915     0.412       0.038
# 19  SGD feature_set_4 StdScale              6     0.948     0.914      0.948      0.914     0.329       0.037
# 38    SGD feature_set_1 MaxAbs              7     0.928     0.869      0.928      0.869     0.200       0.039
# 39    SGD feature_set_3 MaxAbs              8     0.928     0.869      0.928      0.869     0.226       0.040
# 40    SGD feature_set_4 MaxAbs              6     0.928     0.869      0.928      0.869     0.201       0.037
# 41  SGD feature_set_2 StdScale              7     0.928     0.869      0.928      0.868     0.394       0.037
# 42    SGD feature_set_5 MaxAbs              4     0.928     0.868      0.928      0.868     0.173       0.038
# 43           SGD feature_set_1              7     0.928     0.868      0.928      0.868     0.502       0.048
# 44    SGD feature_set_2 MaxAbs              7     0.928     0.868      0.928      0.868     0.223       0.037
# 45  SGD feature_set_5 StdScale              4     0.928     0.868      0.928      0.868     0.291       0.037