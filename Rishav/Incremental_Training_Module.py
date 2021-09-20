import pandas as pd
import os
import datetime
from sklearn.model_selection import cross_val_score

base_path = os.path.dirname(os.path.realpath(__file__))

print('\n{}\tReading dataset: base_data_dev_3m.csv ...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

def prepare_data(in_filename, target='conversion_status'):
    """
    Preparing the data for training, splitting into 
    features and label.
    """
    df_final = pd.read_csv(os.path.join(base_path, in_filename))
    df_final_train = df_final.drop(target, axis=1)
    target = df_final[target]
    
    return df_final_train, target

##########################################

evaluation = {'model': [],
              'feature_count': [],
              'features': [],
              'BAC': []
              } 

def model_evaluation(model, fc, f, bac):
    """
    To keep track of the models used and their
    respective characterisitics with evaluation scores.
    """
    evaluation['model'].append(model)
    evaluation['fc'].append(fc)
    evaluation['f'].append(f)
    evaluation['bac'].append(bac)
    
    df_evaluation = pd.DataFrame(evaluation, columns=['model_name', 'feature_count', 'features', 'BAC'])
    return df_evaluation.sort_values(bt='BAC', ascending=False).round(3)

#############################################


def fit_algorithm(algorithm, in_filename='base_data_dev_3m.csv', cv=5):
    """
    Accepts data from the prepare data function and 
    return balanced accuracy scores of the predicted model.
    """
    X, label = prepare_data(in_filename)
    scores = cross_val_score(algorithm, X, label, cv=cv, scoring='balanced_accuracy')
    
    return scores