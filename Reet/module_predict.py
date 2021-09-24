# this module will be called by other modules and this should return the list of top 250
# customers that are interested to buy the premium services based on various factors

import joblib
import csv
import datetime

def predict_cp(data_stream, prediction_date):
    """
    Function to generate the prediction report for a given date
    :param data_stream: the object that lets us retrieve the input data, data type: module_dep.Datastream object
    :param prediction_date: the date for which the prediction report is requested, data type: datetime.date object
    :return: dataframe consisting of the top 250 potential customers
    """
    # importing the model, change the name if required

    df_input = data_stream.get_data(prediction_date)
    model = joblib.load('cp_model.pkl')
    X = df_input.drop(columns=['date', 'email', 'conversion_status'])

    # drop other columns if reqd and apply any reqd transformations to X

    y_hat = model.predict_proba(X)

    # col 0 has the probab of not converting (0), so including only col 1 in prediction report

    df_prediction_report = df_input[['email']]
    df_prediction_report['conversion_probability'] = y_hat[:, 1]

    # sorting the report in descending order

    df_prediction_report.sort_values(by='conversion_probability', ascending=False)

    # clipping the required 250 entries

    df_prediction_report = df_prediction_report.iloc[:250, :]

    filename_prediction_report = 'prediction_report_' + datetime.datetime.strftime(prediction_date, '%Y%m%d') + '.csv'
    
    df_prediction_report.to_csv(filename_prediction_report, encoding='utf-8', index=False, qouting=csv.QUOTE_ALL)
    
    return df_prediction_report




