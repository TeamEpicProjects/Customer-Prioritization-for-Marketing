import random
import pandas as pd
import numpy as np
import module_PvA
import module_predict
random.seed(23)
from module_dep import DataStream

data_stream = DataStream()

print("Initialize data...")
data_stream.initialize_data()
print("Testing models on 7 random days ...")

dates = pd.Series(pd.date_range(start='2021-05-01', end='2021-07-31'))
random_dates = random.choices(dates.apply(lambda x: x.date()), k=14)
bac_new, rec_new, con_per_new = [], [], []

for prediction_date in random_dates:    
    print(f"Report evaluation for ...{prediction_date}")
    df_p_new = module_predict.predict_cp(data_stream, prediction_date)
    eval_results = module_PvA.evaluate_report(data_stream, prediction_date)
    
    bac_new.append(eval_results[0])
    rec_new.append(eval_results[1])
    con_per_new.append(eval_results[2])

df_eval = pd.DataFrame({'date': random_dates,
                        'BAC': bac_new,
                        'REC': rec_new,
                        'conversion%': con_per_new
                        })
    
print(df_eval.to_string())
print('SDC_f1_s_jlib.pkl')
print()
print(f"BAC mean: {np.mean(bac_new)}")
print(f"REC mean: {np.mean(rec_new)}")
print(f"Conversion % : {np.mean(con_per_new)}")

print(f"BAC var: {np.var(bac_new)}")
print(f"REC var: {np.var(rec_new)}")
print(f"Conversion % var: {np.var(con_per_new)}")


###############################################################################

#           date       BAC       REC  conversion%
# 0   2021-07-25  0.898204  0.937054    91.017964
# 1   2021-07-27  0.939394  0.952050    95.757576
# 2   2021-07-22  0.921053  0.940118    94.736842
# 3   2021-05-08  0.831325  0.896432    86.746988
# 4   2021-06-24  0.961240  0.968223    98.449612
# 5   2021-06-08  0.924528  0.934792    95.597484
# 6   2021-06-18  0.852459  0.914511    87.704918
# 7   2021-05-12  0.750000  0.847603    91.666667
# 8   2021-05-18  0.826389  0.907380    83.333333
# 9   2021-06-10  0.937500  0.952357    96.875000
# 10  2021-05-21  0.719697  0.859848    71.969697
# 11  2021-06-11  0.880000  0.928000    90.400000
# 12  2021-05-03  0.827586  0.899507    85.344828
# 13  2021-05-08  0.831325  0.896432    86.746988
# SDC_f1_s_jlib.pkl

# BAC mean: 0.8643357471455043
# REC mean: 0.9167362095956185
# Conversion % : 89.73913550763076
# BAC var: 0.004836682174515071
# REC var: 0.0011291915817369627
# Conversion % var: 44.73146852900635

##############################################################################

#           date       BAC       REC  conversion%
# 0   2021-07-25  0.904192  0.940048    91.616766
# 1   2021-07-27  0.927273  0.945989    94.545455
# 2   2021-07-22  0.914474  0.941931    93.421053
# 3   2021-05-08  0.807229  0.884384    84.337349
# 4   2021-06-24  0.961240  0.964091    99.224806
# 5   2021-06-08  0.924528  0.934792    95.597484
# 6   2021-06-18  0.819672  0.898117    84.426230
# 7   2021-05-12  0.750000  0.854452    87.500000
# 8   2021-05-18  0.819444  0.903908    82.638889
# 9   2021-06-10  0.929688  0.952549    95.312500
# 10  2021-05-21  0.742424  0.871212    74.242424
# 11  2021-06-11  0.872000  0.924000    89.600000
# 12  2021-05-03  0.827586  0.899507    85.344828
# 13  2021-05-08  0.807229  0.884384    84.337349
# SDC_f4_s_jlib.pkl

# BAC mean: 0.8576413568894365
# REC mean: 0.9142402634058416
# Conversion % : 88.72465236737692
# BAC var: 0.004614088832219501
# REC var: 0.0010502631524212313
# Conversion % var: 41.61607054521216

###############################################################################

#           date       BAC       REC  conversion%
# 0   2021-07-25  0.910180  0.937018    92.814371
# 1   2021-07-27  0.896970  0.919073    92.727273
# 2   2021-07-22  0.881579  0.925483    90.131579
# 3   2021-05-08  0.855422  0.921301    86.746988
# 4   2021-06-24  0.860465  0.913704    89.147287
# 5   2021-06-08  0.874214  0.904140    91.194969
# 6   2021-06-18  0.762295  0.877241    77.049180
# 7   2021-05-12  0.875000  0.834760   150.000000
# 8   2021-05-18  0.847222  0.917797    85.416667
# 9   2021-06-10  0.890625  0.937116    90.625000
# 10  2021-05-21  0.863636  0.924675    87.121212
# 11  2021-06-11  0.856000  0.916000    88.000000
# 12  2021-05-03  0.844828  0.912890    86.206897
# 13  2021-05-08  0.855422  0.921301    86.746988
# SDC_f5_s_jlib.pkl

# BAC mean: 0.862418347524315
# REC mean: 0.9116070333898162
# Conversion % : 92.4234578484555
# BAC var: 0.0011174670030946897
# REC var: 0.0006506598755631913
# Conversion % var: 269.447681524454


############################################################################


#           date       BAC       REC  conversion%
# 0   2021-07-25  0.904192  0.940048    91.616766
# 1   2021-07-27  0.896341  0.930729    91.463415
# 2   2021-07-22  0.881579  0.925483    90.131579
# 3   2021-05-08  0.855422  0.921301    86.746988
# 4   2021-06-24  0.874016  0.916683    91.338583
# 5   2021-06-08  0.874214  0.904140    91.194969
# 6   2021-06-18  0.762295  0.877241    77.049180
# 7   2021-05-12  0.875000  0.834760   150.000000
# 8   2021-05-18  0.847222  0.917797    85.416667
# 9   2021-06-10  0.890625  0.937116    90.625000
# 10  2021-05-21  0.863636  0.924675    87.121212
# 11  2021-06-11  0.856000  0.916000    88.000000
# 12  2021-05-03  0.844828  0.912890    86.206897
# 13  2021-05-08  0.855422  0.921301    86.746988
# SDC_f5_s_t2_jlib.pkl

# BAC mean: 0.8629136599704054
# REC mean: 0.9128688126838692
# Conversion % : 92.40416020359226
# BAC var: 0.0010851882061774455
# REC var: 0.0006843851070997518
# Conversion % var: 268.8595216057114