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
    # If the report for the day is not generated return this message
    if not os.path.isfile(filename_prediction_report):
        return "Prediction report for the selected date doesn't exist."
    
    # Getting the actual and predicted data
    df_actuals = data_stream.get_data(prediction_date)
    df_predicted = pd.read_csv(os.path.join(base_path, filename_prediction_report))
    
    # Merging the actual and predicted files on email
    df_actuals_predicted = pd.merge(df_actuals, df_predicted, on='email', how='inner')
    
    # Creating a DataFrame to store the PvA report
    df_display = pd.DataFrame()

    # Converting the probabilities into binary choices based on the threshold 1,0
    df_display['conversion_status_predicted'] = pd.Series(np.where(df_actuals_predicted['conversion_probability']>=0.5, 1, 0), dtype=np.int)
    y_pred = df_display['conversion_status_predicted'].to_numpy()

    # Extracting actual conversion_status from the merged model
    y_true = df_actuals_predicted['conversion_status'].to_numpy()
    # Storing the extracted y's into the new DataFrame that we created
    df_display['conversion_status_actual'] = pd.Series(y_true)

    # Creating a confusion matrix for actual v/s predicted
    df_confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    # Calculating the Balanced Accuracy and Recall Scores
    bac_score = bac(y_true, y_pred)
    rc_score = rc(y_true, y_pred)
    
    # Calculating the conversion_rate
    conversion_predicted = df_display['conversion_status_actual'].sum()
    conversion_actual = df_display['conversion_status_predicted'].sum()
    conversion_rate = (conversion_actual/conversion_predicted)*100
    
    return rc_score, bac_score, conversion_rate

###################################################################################
