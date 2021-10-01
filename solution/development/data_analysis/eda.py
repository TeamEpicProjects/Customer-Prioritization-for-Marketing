# Steps :- 
# 1. Check distribution of various features
# 2. Check for correlation values
# 3. Check for outliers
# 4. Remove outliers
# 5. Check for correlation values
# 6. Choose important features

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
# get_ipython().run_line_magic('matplotlib', 'inline')

df_final = pd.read_csv('base_data_dev_3m.csv')
df_final.head(3)



date = df_final['date'].copy()
df_final.drop(columns=['date', 'email'], axis=1, inplace=True)
df_final.shape

# Function to make distribution and count plots
# df_final = df_final.sample(frac=0.35)

def plot_(data, col, figsize=(20,2), hist=False):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    if hist:
        sns.histplot(data=data, x=col)
        ax.set_yscale('log')
        ax.set_xscale('log')
    else:
        sns.countplot(y=col, data=data)


plot_(df_final, 'nunique_gender')
plt.savefig('base_data_dev_3m_countplot_nunique_gender.png')


plot_(df_final, 'conversion_status')
plt.savefig('base_data_dev_3m_countplot_conversion_status.png')

columns = df_final.columns.tolist()

fig, ax = plt.subplots(2, 4)
fig.set_size_inches(16, 18)
sns.boxplot(x='conversion_status', y=columns[0], data=df_final, ax=ax[0, 0])
sns.boxplot(x='conversion_status', y=columns[1], data=df_final, ax=ax[0, 1])
sns.boxplot(x='conversion_status', y=columns[2], data=df_final, ax=ax[0, 2])
sns.boxplot(x='conversion_status', y=columns[3], data=df_final, ax=ax[0, 3])
sns.boxplot(x='conversion_status', y=columns[4], data=df_final, ax=ax[1, 0])
sns.boxplot(x='conversion_status', y=columns[5], data=df_final, ax=ax[1, 1])
sns.boxplot(x='conversion_status', y=columns[6], data=df_final, ax=ax[1, 2])
sns.boxplot(x='conversion_status', y=columns[7], data=df_final, ax=ax[1, 3])
plt.savefig('base_data_dev_3m_boxplots.png')

# def remove_outlier_IQR(df, n=4):
#     Q1=df.quantile(0.25)
#     Q3=df.quantile(0.75)
#     IQR=Q3-Q1
#     df_final = df[~(df > (Q3 + n * IQR))]
#     return df_final

df_final_or = df_final[(df_final.count_user_stay <= 50) & (df_final.count_pay_attempt <= 10) & (df_final.count_buy_click <= 25) &
                        (df_final.nunique_gender <= 2) & (df_final.nunique_dob <= 5) & (df_final.nunique_report_type <= 2) &
                        (df_final.nunique_language <= 2)]
print(df_final_or.head())


fig, ax = plt.subplots(2, 4)
fig.set_size_inches(16, 18)
sns.boxplot(x='conversion_status', y=columns[0], data=df_final_or, ax=ax[0, 0])
sns.boxplot(x='conversion_status', y=columns[1], data=df_final_or, ax=ax[0, 1])
sns.boxplot(x='conversion_status', y=columns[2], data=df_final_or, ax=ax[0, 2])
sns.boxplot(x='conversion_status', y=columns[3], data=df_final_or, ax=ax[0, 3])
sns.boxplot(x='conversion_status', y=columns[4], data=df_final_or, ax=ax[1, 0])
sns.boxplot(x='conversion_status', y=columns[5], data=df_final_or, ax=ax[1, 1])
sns.boxplot(x='conversion_status', y=columns[6], data=df_final_or, ax=ax[1, 2])
sns.boxplot(x='conversion_status', y=columns[7], data=df_final_or, ax=ax[1, 3])
plt.savefig('base_data_dev_3m_boxplots_outlier_removed.png')

plt.figure(figsize=(10,10))
sns.heatmap(df_final.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.savefig('base_data_dev_3m_correlation.png')

plt.figure(figsize=(10,10))
sns.heatmap(df_final_or.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.savefig('base_data_dev_3m_correlation_outlier_removed.png')

# df_final_or.to_csv('base_data_dev_3m_or_.csv', encoding='utf-8', index=False)
# Final Dataset ready for Machine Learning

df_ = pd.read_csv('/home/azureuser/packtgp1/base_data_resampled_tomek.csv')
df_.head(3)
print(df_.shape)

columns = df_.columns
print("Columns of base_data_resampled_tomek: ", columns)


# Checking outliers

fig, ax = plt.subplots(2, 4)
fig.set_size_inches(16, 18)
sns.boxplot(x='conversion_status', y=columns[0], data=df_, ax=ax[0, 0])
sns.boxplot(x='conversion_status', y=columns[1], data=df_, ax=ax[0, 1])
sns.boxplot(x='conversion_status', y=columns[2], data=df_, ax=ax[0, 2])
sns.boxplot(x='conversion_status', y=columns[3], data=df_, ax=ax[0, 3])
sns.boxplot(x='conversion_status', y=columns[4], data=df_, ax=ax[1, 0])
sns.boxplot(x='conversion_status', y=columns[5], data=df_, ax=ax[1, 1])
sns.boxplot(x='conversion_status', y=columns[6], data=df_, ax=ax[1, 2])
sns.boxplot(x='conversion_status', y=columns[7], data=df_, ax=ax[1, 3])
plt.savefig('base_data_resampled_tomek_boxplots.png')


# Checking correlaion analysis

plt.figure(figsize=(10,10))
sns.heatmap(df_.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.savefig('base_data_resampled_tomek_coorelation.png')

# Feature selection

feature_set_1 = columns.tolist().remove('count_user_stay')    # removing multicollinear feature
feature_set_2 = columns.tolist().remove('nunique_language')  # removing feature which is least corelated with target
feature_set_3 = [e for e in columns.tolist() if e not in ('count_user_stay', 'count_sessions')]       # removing multicollinear features from top 2 highest cor values
feature_set_4 = [e for e in columns.tolist() if e not in ('count_user_stay', 'count_sessions', 'nunique_beacon_type')]

print(feature_set_1)
print(feature_set_2)
print(feature_set_3)
print(feature_set_4)
