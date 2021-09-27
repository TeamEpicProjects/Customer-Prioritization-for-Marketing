import pandas as pd

a = ['phone', 'status', 'call_log']
b = ['uuid', 'beacon_type', 'beacon_value', 'log_date', 'Log_date']
c = ['id', 'email', 'primary_phone', 'profile_submit_count']
ct = ['id', 'cid', 'amount', 'status']
s = ['uuid', 'phone', 'gender', 'dob', 'language', 'email', 'report_type', 'device']
tp = ['ctid', 'variant', 'language']


columns = (a, b, c, ct, s, tp)
names = ['a', 'b', 'c', 'ct', 's', 'tp']
lists, features, check = [], [], []

for index1, first_column_name in enumerate(columns):
    for index2, second_column_name in enumerate(columns):
        if first_column_name==second_column_name:
            continue
        else:
            # Getting the common column names 
            common = set(first_column_name)&set(second_column_name)
            
            # Checking if k is empty or if it's already counted
            if len(common)<=0 or names[index1] + names[index2] in check:
                continue
            else:
                features.append(common)
                lists.append({names[index1], names[index2]})
                check.append(names[index1] + names[index2])
                check.append(names[index2] + names[index1])

        
if len(lists)==len(features):
    df = pd.DataFrame(data={'files': lists, 'features': features})
    # df.to_excel('similar.xlsx')
print(df)

##############################################################################


#      files    features
# 0  {ct, a}    {status}
# 1   {s, a}     {phone}
# 2   {s, b}      {uuid}
# 3  {ct, c}        {id}
# 4   {s, c}     {email}
# 5  {s, tp}  {language}