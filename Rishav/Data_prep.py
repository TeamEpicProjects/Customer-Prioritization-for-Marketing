import pandas as pd
import numpy as np

dfa = pd.read_csv('a(61).csv')
dfb = pd.read_csv('b(1.9).csv')
dfc = pd.read_csv('c(75).csv')
dfct = pd.read_csv('ct(204).csv')
dfs = pd.read_csv('s(851).csv')
dftp = pd.read_csv('tp(107).csv')

def get_info(df, include_unique=True):
    """Function to get the summary of
    the entire dataframe"""
    
    column = [col for col in df.columns]
    col_type = [type(cell) for cell in df.loc[0,:]]
    null_count = [df[col].isna().sum() for col in df.columns]
    null_percent = [(df[col].isna().sum()/df.shape[0])*100 for col in df.columns]
    
    if include_unique:
        unique = [df[col].unique() for col in df.columns]
        unique_count = [df[col].nunique() for col in df.columns]
        df_info = pd.DataFrame({'column': column, 'column_type': col_type,
                                'null_count': null_count, 'unique_count': unique_count,
                                'null_percent': null_percent, 'unique_values': unique})
    else:
        df_info = pd.DataFrame({'column': column, 'column_type': col_type,
                                'null_count': null_count, 'null_percent': null_percent})
    return df_info


def codes_a1(x):
    """creating first set of category codes
    for table a"""
    
    if x in ['purchase', 'Already purchased', 'Converted']:
        return 1
    elif x in ['User is Interested', 'New product potential', 'Partially interested']:
        return 2
    else:
        return 0
    
def codes_a2(x):
    """creating second set of category
    codes for table a"""
    if x>0 and x<2:
        return 1
    else:
        return x    
    
def codes_s1(x):
    """category codes for table s"""
    if x=='E':
        return 'ENG'
    elif x in ['H', 'HINDI']:
        return 'HIN'
    elif x=='M':
        return 'MAR'
    else:
        return x

def codes_ct1(x):
    """category codes for table ct1"""
    if x=='N':
        return 0
    else:
        return 1   
    
def lang_codes(x):
    """fixing the inconsistencies with language names"""
    if x in ['eng', 'Eng', 'ENG', 'en', '--', '0', 'nil']:
        return 'Eng'
    elif x in ['hin', 'Hin', 'Hi']:
        return 'Hin'
    elif x in ['tam', 'Tam', 'TAM']:
        return 'Tam'
    elif x in ['mal', 'MAL', 'Mal']:
        return 'Mal'
    elif x in ['tel', 'Tel']:
        return 'Tel'
    elif x in ['Kan', 'kan']:
        return 'Kan'
    elif x in ['mar', 'MAR', 'Mar']:
        return 'Mar'
    elif x in ['Ben', 'ben', 'BEN']:
        return 'Ben'
    elif x in ['ori', 'Ori', 'ORI']:
        return 'Ori'
    elif x in ['Guj', 'guj', 'GUJ']:
        return 'Guj'
    else:
        return 'Sin'
    
# Cleaning table a data    
    
dfa_clean = dfa.drop(columns=['product', 'pay_mode', 'marker', 'type'], axis=1)
dfa_clean['phone']= dfa_clean.phone.astype(str)    
dfa_clean.status = dfa_clean.status.apply(lambda x: codes_a1(x))
dfa_clean['date'] = dfa_clean['log_time'].apply(lambda x: x[:11])
dfa_clean.drop(columns=['log_time'], axis=1, inplace=True)
dfa_grouped = dfa_clean.groupby(['date', 'phone'])    
dfa_grouped = dfa_grouped.mean().add_suffix('_').reset_index()
dfa_grouped['status'] = dfa_grouped['status_'].apply(lambda x: codes_a2(x))
dfa_final = dfa_grouped.drop(columns=['status_'], axis=1)

    
# Cleaning table b data
dfb_clean = dfb.drop(columns=['status'], axis=1)
dfb_clean.uuid = dfb_clean.uuid.astype(str)
dfb_clean['date'] = dfb_clean['log_date'].apply(lambda x: x[:11])
dfb_clean.drop('log_date', axis=1, inplace=True)
dfb_clean.uuid = dfb_clean.uuid.apply(lambda x: x[:-2])
dfb_grouped = dfb_clean.groupby(['date', 'uuid']) 
dfb_final = dfb_grouped.agg({'beacon_value': np.sum, 'beacon_type': 'count'}).add_suffix('_').reset_index()

# Cleaning table c data
dfc_clean = dfc.copy()
dfc_clean['secondary_phones'] = dfc_clean['secondary_phones'].astype(str)
dfc_clean.primary_phone = dfc_clean['primary_phone'].astype(str)
dfc_clean.id = dfc_clean.id.astype(str)

# Cleaning table s data
dfs_clean = dfs.drop(columns=['status', 'log_date'], axis=1)
dfs_clean['date'] = dfs['log_date'].apply(lambda x:x[:11])
dfs_clean.phone = dfs_clean.phone.astype(str)
dfs_clean.email = dfs_clean.email.astype(str)
dfs_clean.language = dfs_clean['language'].apply(lambda x: codes_s1(x))
dfs_clean['device'] = dfs_clean['device'].map({'mobile': 'Mobile', 'pc': 'PC',
                                               'desktop': 'PC','MOBILE': 'Mobile'})
dfs_clean = dfs_clean.astype({'gender': 'category', 'report_type': 'category', 'device': 'category', 'uuid': str})


# Cleaning table ct data
dfct_clean = dfct
dfct_clean['timestamp'] = dfct_clean.timestamp.apply(lambda x:x[:11])
dfct_clean.id = dfct_clean.id.astype(str)
dfct_clean.cid = dfct_clean.cid.astype(str)
dfct_clean['status'] = dfct_clean.status.apply(lambda x: codes_ct1(x))


# Cleaning table tp data
dftp_clean = dftp.drop(columns=['status'], axis=1)
dftp_clean.variant = dftp_clean.variant.astype('category')
dftp_clean.language = dftp_clean.language.astype('category')
dftp_clean['cid'] = dftp_clean['ctid']
dftp_clean.drop(columns=['ctid'], axis=1, inplace=True)
dftp_clean.cid = dftp_clean.cid.astype(str)
dftp_clean['language'] = dftp['language'].apply(lambda x: lang_codes(x))
dftp_clean.drop_duplicates(inplace=True)

# Merging tables c, ct and tp to create df_ccttp
df_cct = pd.merge(dfc_clean, dfct_clean, on='id', how='inner')
df_ccttp = pd.merge(df_cct, dftp_clean, on='cid', how='inner')

# Merging tables b and s to create dfsb
dfsb = pd.merge(dfb_final, dfs_clean, on='uuid', how='inner')


# Pretty printing the merged tables
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_ccttp.head())
    print(dfsb.head())
    #print(dfb_final.head())
    #print()
    #print(dftp_clean.head())
    #print(get_info(dftp))
    
    