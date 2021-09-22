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

######################

df_models = pd.read_csv('model_scores.csv').iloc[:, 1:]
df_models.columns = ['model_name', 'feature_count', 'bac_test', 'rec_test', 'bac_train', 'rec_train', 'time_fit', 'time_score']

df_models_info = df_info(df_models)
print('\n"models" dataset summary:')
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_models.shape[0], df_models.shape[1], df_models_info[1]))
print(df_models_info[0].to_string())
print('\n"models" dataset head:')
print(df_models.to_string())

df_filtered_models = df_models.loc[(df_models.bac_train < 0.96) & (df_models.rec_train < 0.97) & (df_models.bac_test >= 0.92) & (df_models.rec_test >= 0.87)]

print('\n"filtered_models" dataset summary:')
print('\t{} rows x {} columns'.format(df_filtered_models.shape[0], df_filtered_models.shape[1]))
print(df_filtered_models.to_string())

######################

# "models" dataset summary:
# 	39 rows x 8 columns | 0.01 MB approx memory usage
#         col_name                 col_type  null_count  nunique
# 0     model_name            <class 'str'>           0       35
# 1  feature_count    <class 'numpy.int64'>           0        4
# 2       bac_test  <class 'numpy.float64'>           0       11
# 3       rec_test  <class 'numpy.float64'>           0       15
# 4      bac_train  <class 'numpy.float64'>           0       15
# 5      rec_train  <class 'numpy.float64'>           0       17
# 6       time_fit  <class 'numpy.float64'>           0       39
# 7     time_score  <class 'numpy.float64'>           0       15
#
# "models" dataset head:
#                     model_name  feature_count  bac_test  rec_test  bac_train  rec_train  time_fit  time_score
# 0    GB feature_set_1 StdScale              7     0.988     0.998      0.989      0.999    12.657       0.063
# 1             GB feature_set_1              7     0.988     0.998      0.989      0.999    14.884       0.112
# 2      GB feature_set_1 MaxAbs              7     0.988     0.998      0.989      0.999    12.639       0.063
# 3              RF all features             13     0.986     0.994      0.996      0.999     9.485       0.372
# 4             RF feature_set_1              7     0.986     0.994      0.996      0.999    10.112       0.500
# 5    GB feature_set_4 StdScale              6     0.986     0.999      0.986      1.000     8.578       0.064
# 6      GB feature_set_4 MaxAbs              6     0.986     0.999      0.986      1.000     8.590       0.064
# 7    GB feature_set_3 StdScale              8     0.986     0.999      0.986      0.999     9.420       0.063
# 8      GB feature_set_3 MaxAbs              8     0.986     0.999      0.986      0.999     9.458       0.064
# 9            SGD feature_set_1              7     0.984     0.995      0.985      0.995     0.268       0.049
# 10   RF feature_set_3 StdScale              8     0.984     0.994      0.991      0.999     6.882       0.349
# 11     RF feature_set_3 MaxAbs              8     0.984     0.994      0.991      0.999     6.897       0.349
# 12     RF feature_set_4 MaxAbs              6     0.984     0.994      0.991      0.999     6.776       0.336
# 13   RF feature_set_4 StdScale              6     0.984     0.994      0.991      0.999     6.767       0.334
# 14            SGD all features             13     0.982     0.988      0.982      0.988     0.352       0.063
# 15             LR all features             13     0.963     0.943      0.962      0.942    27.399       0.038
# 16            LR feature_set_1              7     0.962     0.942      0.962      0.941     4.919       0.066
# 17  SGD feature_set_1 StdScale              7     0.949     0.915      0.950      0.916     0.341       0.038
# 18  SGD feature_set_3 StdScale              8     0.949     0.914      0.949      0.915     0.412       0.037
# 19  SGD feature_set_4 StdScale              6     0.948     0.914      0.948      0.914     0.333       0.037
# 20   LR feature_set_4 StdScale              6     0.941     0.898      0.941      0.898     0.472       0.037
# 21   LR feature_set_1 StdScale              7     0.941     0.897      0.941      0.897     0.576       0.038
# 22   LR feature_set_3 StdScale              8     0.941     0.898      0.941      0.898     0.535       0.037
# 23     LR feature_set_3 MaxAbs              8     0.929     0.872      0.929      0.872     1.182       0.038
# 24     LR feature_set_4 MaxAbs              6     0.929     0.872      0.929      0.872     0.974       0.038
# 25     LR feature_set_1 MaxAbs              7     0.929     0.871      0.929      0.871     1.095       0.038
# 26            GB feature_set_1              7     0.928     0.872      0.930      0.873     6.793       0.074
# 27   GB feature_set_2 StdScale              7     0.928     0.872      0.930      0.873     6.958       0.063
# 28     GB feature_set_2 MaxAbs              7     0.928     0.872      0.930      0.873     6.971       0.063
# 29            LR feature_set_1              7     0.928     0.871      0.929      0.871     1.178       0.049
# 30   LR feature_set_2 StdScale              7     0.928     0.871      0.929      0.871     0.372       0.038
# 31     LR feature_set_2 MaxAbs              7     0.928     0.871      0.928      0.871     1.262       0.038
# 32    SGD feature_set_3 MaxAbs              8     0.928     0.869      0.928      0.869     0.237       0.037
# 33    SGD feature_set_1 MaxAbs              7     0.928     0.869      0.928      0.869     0.202       0.038
# 34    SGD feature_set_4 MaxAbs              6     0.928     0.869      0.928      0.869     0.208       0.038
# 35  SGD feature_set_2 StdScale              7     0.928     0.869      0.928      0.868     0.406       0.039
# 36           SGD feature_set_1              7     0.928     0.868      0.928      0.868     0.522       0.049
# 37    SGD feature_set_2 MaxAbs              7     0.928     0.868      0.928      0.868     0.228       0.038
# 38            RF feature_set_1              7     0.928     0.872      0.931      0.875     5.178       0.326
#
# "filtered_models" dataset summary:
# 	16 rows x 8 columns
#                     model_name  feature_count  bac_test  rec_test  bac_train  rec_train  time_fit  time_score
# 17  SGD feature_set_1 StdScale              7     0.949     0.915      0.950      0.916     0.341       0.038
# 18  SGD feature_set_3 StdScale              8     0.949     0.914      0.949      0.915     0.412       0.037
# 19  SGD feature_set_4 StdScale              6     0.948     0.914      0.948      0.914     0.333       0.037
# 20   LR feature_set_4 StdScale              6     0.941     0.898      0.941      0.898     0.472       0.037
# 21   LR feature_set_1 StdScale              7     0.941     0.897      0.941      0.897     0.576       0.038
# 22   LR feature_set_3 StdScale              8     0.941     0.898      0.941      0.898     0.535       0.037
# 23     LR feature_set_3 MaxAbs              8     0.929     0.872      0.929      0.872     1.182       0.038
# 24     LR feature_set_4 MaxAbs              6     0.929     0.872      0.929      0.872     0.974       0.038
# 25     LR feature_set_1 MaxAbs              7     0.929     0.871      0.929      0.871     1.095       0.038
# 26            GB feature_set_1              7     0.928     0.872      0.930      0.873     6.793       0.074
# 27   GB feature_set_2 StdScale              7     0.928     0.872      0.930      0.873     6.958       0.063
# 28     GB feature_set_2 MaxAbs              7     0.928     0.872      0.930      0.873     6.971       0.063
# 29            LR feature_set_1              7     0.928     0.871      0.929      0.871     1.178       0.049
# 30   LR feature_set_2 StdScale              7     0.928     0.871      0.929      0.871     0.372       0.038
# 31     LR feature_set_2 MaxAbs              7     0.928     0.871      0.928      0.871     1.262       0.038
# 38            RF feature_set_1              7     0.928     0.872      0.931      0.875     5.178       0.326