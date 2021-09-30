All files needed to deploy on Heroku:

<pre>
templates/base.html               : base template  
templates/index.html              : web app frontend - landing page template for user interface   
base_data_resampled_tomek_ops.csv : dataset for simulating new data on which prediction can be done   
module_dep.py                     : data extraction and preparation module   
module_inc_train.py               : incremental training module to update the model  
module_predict.py                 : prediction module   
module_pva.py                     : predicted vs. actuals (evaluation) module  
Procfile                          : configuration to start app   
requirements.txt                  : dependencies  
SDC_f1_s_jlib.pkl                 : the ML model used for predictions   
web_app_flask.py                  : web app backend  
</pre>
