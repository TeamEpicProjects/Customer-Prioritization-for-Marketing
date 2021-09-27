# This module will be called by other modules and this should return the list of top 250
# customers that are interested to buy the premium services based on various factors

import joblib
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

def predict_cp(data_stream, prediction_date):
    """
    Function to generate the prediction report for a given date
    :param data_stream: the object that lets us retrieve the input data, data type: module_dep.Datastream object
    :param prediction_date: the date for which the prediction report is requested, data type: datetime.date object
    :return: dataframe consisting of the top 250 potential customers
    """

    df_input = data_stream.get_data(prediction_date)
    
    # Loading the model
    model_chosen = 'SDC_f1_s_jlib.pkl'
    model = joblib.load(model_chosen)
    
    # Preparing X's
    X = df_input.drop(columns=['date', 'email', 'conversion_status'])
    
    feature_set_1 = ['transactions_amount', 'count_pay_attempt', 'nunique_beacon_type',
                     'count_user_stay', 'count_buy_click', 'profile_submit_count',
                     'sum_beacon_value']
    feature_set_4 = ['sum_beacon_value', 'count_pay_attempt', 'count_buy_click',
                      'nunique_report_type', 'nunique_device', 'transactions_amount']
    
    feature_set_5 = ['count_pay_attempt', 'count_buy_click',
                      'nunique_report_type', 'profile_submit_count']
        

    X_scaled = StandardScaler().fit_transform(X[feature_set_1])
    # Getting the probabilities of X_scaled
    y_hat = model.predict_proba(X_scaled)
    # Creating the prediction report with email and conversion_probability
    df_prediction_report = df_input[['email']]
    df_prediction_report['conversion_probability'] = y_hat[:, 1]
    df_prediction_report.sort_values(by='conversion_probability', ascending=False, inplace=True)

    # Filtering the Top-250 entries
    df_prediction_report = df_prediction_report.iloc[:250, :]
    
    # Creating a csv of the report
    filename_prediction_report = 'prediction_report_' + datetime.datetime.strftime(prediction_date, '%Y%m%d') + '.csv'
    df_prediction_report.to_csv(filename_prediction_report, encoding='utf-8', index=False)

    return df_prediction_report