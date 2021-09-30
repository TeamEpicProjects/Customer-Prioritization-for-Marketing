from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms.fields.html5 import DateField
from wtforms.validators import DataRequired
from wtforms import validators, RadioField, SubmitField
import module_pva
import module_dep
import module_predict
import module_inc_train
import pandas as pd
import os

base_path = os.path.dirname(os.path.realpath(__file__))

data_stream = module_dep.DataStream()
data_stream.initialize_data()
app = Flask(__name__)

app.config['SECRET_KEY'] = '#$%^&*'

class InfoForm(FlaskForm):
    report_type = RadioField('Report Type', choices=[('Prediction Report','Prediction Report'),('Predicted v/s Actual Report','Predicted v/s Actual report')])
    report_date = DateField('Report Date', format='%Y-%m-%d', validators=(validators.DataRequired(),))
    submit = SubmitField('Submit')

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def index():
    form = InfoForm()
    df_prediction_report = pd.DataFrame()
    
    if form.validate_on_submit():

        if form.report_type.data=='Prediction Report':
            df_prediction_report = module_predict.predict_cp(data_stream, form.report_date.data)
            df_prediction_report.sort_values(by='conversion_probability', ascending=False, inplace=True)
            df_prediction_report.reset_index(drop=True, inplace=True)
            form.report = ""
            print("Incremental training begins...")
            module_inc_train.inc_train(data_stream, form.report_date.data)
            return render_template('index.html', form=form, tables=[df_prediction_report.to_html(classes='data')], titles=df_prediction_report.columns.values)

        else:
            pva_report = module_pva.evaluate_report(data_stream, form.report_date.data)
            if type(pva_report)==str:
                df_pva_report = pd.DataFrame({'': [pva_report]})
                #form.report = form_report
            else:
                df_pva_report = pd.DataFrame({'': [str(round(pva_report[0], 4)), str(round(pva_report[1], 4)), str(pva_report[2]), str(pva_report[3]), str(round(pva_report[4], 4))]},
                                                   index=['Balanced Accuracy', 'Recall', 'Conversion actual', 'Coversion Predicted', 'Conversion Ratio'])
                                          
            return render_template('index.html', form=form, tables=[df_pva_report.to_html(classes='data')], titles=df_pva_report.columns.values)
    
    else:
        print(form.errors)
        return render_template('index.html', form=form, tables=[df_prediction_report.to_html(classes='data')], titles=df_prediction_report.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    