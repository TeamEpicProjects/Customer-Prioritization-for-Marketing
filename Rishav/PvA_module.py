# This script creates a report of Actual v/s Predicted conversion_status

import datetime
import os
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score as rc
from sklearn.metrics import balanced_accuracy_score as bac

def evaluate_report(data_stream, prediction_date):
    """
    Function to get the prediction report of a specific date.
    :param data_stream: the object that lets us retrieve the input data, data type: module_dep.Datastream object
    :param prediction_date: the date for which the prediction report was generated, data type: datetime.date object
    :return: DataFrame consisting of the PvA report
    """
    base_path = os.path.dirname(os.path.realpath(__file__))
    filename_prediction_report = 'prediction_report_' + datetime.datetime.strftime(prediction_date, '%Y%m%d') + '.csv'
    df_input = data_stream.get_data(prediction_date)
    df_predict = pd.read_csv(os.path.join(base_path, filename_prediction_report))
    y_pred = df_predict['conversion_probability'].to_numpy()
    y_true = df_input['conversion_status'].to_numpy()
    df_confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    bac_score = bac(y_true, y_pred)
    rc_score = rc(y_true, y_pred)
    
    df_display = pd.DataFrame()
    df_display['conversion_status_actual'] = pd.Series(y_true)
    df_display['conversion_status_prdicted'] = pd.Series(np.where(y_pred>0.5, 1, 0))
    conversion_predicted = df_display['conversion_status_actual'].sum()
    conversion_actual = df_display['conversion_status_predicted'].sum()
    correct_percent = conversion_predicted/conversion_actual
    
    
    return f"{df_confusion}\nBAC score:\t{bac_score}, RC score:\t{rc_score}\n\
            Predicted conversions:\t{conversion_predicted}, Actual Conversions:\t{conversion_actual}\
            \nConversion% :\t{correct_percent}"
    
###################################################################################




# import matplotlib.pyplot as plt


# def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
#     plt.matshow(df_confusion, cmap=cmap) # imshow
#     #plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(df_confusion.columns))
#     plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#     plt.yticks(tick_marks, df_confusion.index)
#     #plt.tight_layout()
#     plt.ylabel(df_confusion.index.name)
#     plt.xlabel(df_confusion.columns.name)



























import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler 


def df_info(df):
    """
    Function to get information about a dataframe
    """
    col_name_list = list(df.columns)
    col_type_list = [type(col) for col in df.iloc[0, :]]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_memory_usage_list = [df[col].memory_usage(deep=True) for col in col_name_list]
    df_total_memory_usage = sum(col_memory_usage_list) / 1048576
    return pd.DataFrame({'col_name': col_name_list, 'col_type': col_type_list, 'null_count': col_null_count_list, 'nunique': col_unique_count_list}), df_total_memory_usage


# Reading the dataset base_data_dev_3m.csv
base_path = os.path.dirname(os.path.realpath(__file__))
in_file_name = "base_data_dev_3m.csv"
print('\n{}\tReading dataset: {} ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), in_file_name))
df_eval = pd.read_csv(os.path.join(base_path, in_file_name))
df_eval_info = df_info(df_eval)
print('\n{}\t"base_data_dev_3m" dataset summary:'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print('\t{} rows x {} columns | {:.2f} MB approx memory usage'.format(df_eval.shape[0], df_eval.shape[1], df_eval_info[1]))
print(df_eval_info[0].to_string())
print('\n"base_data_dev_3m_or" dataset head:')
print(df_eval.head().to_string())

# Seperating the features and labels
print("\nSplitting into features and lables ...")
X = df_eval.drop(columns=['conversion_status', 'email', 'date'], axis=1)
y = df_eval['conversion_status']
# Scaling X
X_scaled = StandardScaler().fit_transform(X)





###############################################################################