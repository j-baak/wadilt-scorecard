# import packages
import os, boto3, datetime
import pandas as pd
import numpy as np
import plotly.express as px
pd.options.plotting.backend = "plotly"
import streamlit as st  # 🎈 data web app development
from scipy import stats


# define local data path
local_dir = './' #define_datafolder()
local_dir_json = local_dir + 'json/'


# define target data path
target_data_path = './target/backadjusted_target_K200f.csv'


# define S3 data path
bucket_name = 'bucket-4-clients'
s3_folder = 'predictions_K200f'
s3_data_path = 'predictions_K200f/prediction_K200f_'


# get list of saved predicitons in json
local_json_list = sorted([file for file in os.listdir(local_dir_json) if file.endswith('.json')])


# download json from AWS
s3 = boto3.resource('s3') # assumes credentials & configuration are in ./streamlit/secrets.toml file
bucket = s3.Bucket(bucket_name)
for obj in bucket.objects.filter(Prefix=s3_folder):
    if obj.key[-1] == '/': continue  # pass subfolder in AWS
    if os.path.relpath(obj.key, s3_folder) in local_json_list: 
        print('pass: ', os.path.relpath(obj.key, s3_folder))
        continue 
    else:
        if 'json' in obj.key:
            target = os.path.join(local_dir_json, os.path.relpath(obj.key, s3_folder))
            bucket.download_file(obj.key, target)
        
        
# read in data and compile in df
df_pred_raw = pd.DataFrame()
local_json_list = sorted([file for file in os.listdir(local_dir_json) if file.endswith('.json')])
for dd in local_json_list:
    tmp = pd.read_json(local_dir_json + dd).loc[['pred'], :]
    tmp['date'] = datetime.datetime.strptime(dd.split('_')[2].split('.')[0], "%Y%m%d").date()
    df_pred_raw = pd.concat([df_pred_raw, tmp])
    
df_pred = pd.DataFrame()
for row in df_pred_raw.iterrows():    
    tmp = row[1].to_frame().T   
    # switch 0925 and 1030 for data before 20220214
    if tmp['date'][0] < datetime.date(2022, 2, 14):
        tmp.rename({'K200f.1030': 'K200f.0925', 'K200f.0925': 'K200f.1030'}, axis='columns', inplace=True)    
    df_pred = pd.concat([df_pred, tmp])
    
df_pred.set_index('date', inplace=True)
df_pred.index = pd.DatetimeIndex(df_pred.index)    
df_pred = df_pred.loc[:, ['K200f.1030', 'K200f.1130']]


# cast floating type to float
df_pred['K200f.1030'] = df_pred['K200f.1030'].astype('float')
df_pred['K200f.1130'] = df_pred['K200f.1130'].astype('float')


# extract congruent signals
df_pred_cong = df_pred.copy()
df_pred['cong'] = df_pred.iloc[:, 0].mul(df_pred.iloc[:, 1]).apply(np.sign)
df_pred = df_pred.loc[df_pred['cong'] > 0, ['K200f.1030', 'K200f.1130']]
assert(df_pred['K200f.1030'].apply(np.sign).equals(df_pred['K200f.1130'].apply(np.sign)))
df_pred = df_pred[['K200f.1030']]  # final, congruent signal
df_pred.columns = ['signal']


# read in target data
df_ground = pd.read_csv(target_data_path, index_col='date', parse_dates=True).loc[:, ['O2MC1030', 'mrng_mean_PCT']]
df_ground = df_ground.loc[df_ground.index.isin(df_pred.index), :]
df_pred = df_pred.loc[df_pred.index.isin(df_ground.index), :]  # remove holidays from df_pred
assert(sum(df_pred.index == df_ground.index) == len(df_pred))


# hit/miss
# newidx = ['K200f.1030', 'K200f.1130']
df_hits = pd.DataFrame()
df_hits['K200f.1030'] = np.sign(df_pred['signal'].values * df_ground['O2MC1030'])
df_hits['K200f.1130'] = np.sign(df_pred['signal'].values * df_ground['mrng_mean_PCT'])
df_hits['rows'] = list(range(1, len(df_hits) + 1))


# summary stats
hitrate_1030 = np.sum(df_hits['K200f.1030'] > 0) / len(df_hits)
pvalue_1030 = stats.binom_test(
    df_hits['K200f.1030'].replace({-1: 0}).sum(), 
    n=len(df_hits), p=0.5, alternative='greater') 
hitrate_1130 = np.sum(df_hits['K200f.1130'] > 0) / len(df_hits)
pvalue_1130 = stats.binom_test(
    df_hits['K200f.1130'].replace({-1: 0}).sum(), 
    n=len(df_hits), p=0.5, alternative='greater') 

df_summary = pd.DataFrame([[hitrate_1030, hitrate_1130], [pvalue_1030, pvalue_1130]],
                          columns=['K200f.1030', 'K200f.1130'], 
                          index=['hit rate', 'binom p-value']).round(3)


# rolling hit rates: 5-day
df_roll0 = df_hits.drop(columns=['rows'])
window_size = 5
df_roll5 = df_roll0.replace({-1: 0}).rolling(window=window_size).sum() / window_size
df_roll5['refline'] = 0.5
fig5 = df_roll5.plot(kind='line') #, title='5-day rolling accuracy') 
fig5.update_layout(xaxis_title="5-day moving average")
fig5.update_xaxes(showgrid=False) # True, gridwidth=0.1)#, gridcolor='Grey')
fig5.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')#gridcolor='Grey')
# fig5.show()

                     
# 10-day
window_size = 10
df_roll10 = df_roll0.replace({-1: 0}).rolling(window=window_size).sum() / window_size
df_roll10['refline'] = 0.5
fig10 = df_roll10.plot(kind='line')
fig10.update_layout(xaxis_title="10-day moving average")
fig10.update_xaxes(showgrid=False)
fig10.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')
# fig10.show()


# 15-day
window_size = 15
df_roll15 = df_roll0.replace({-1: 0}).rolling(window=window_size).sum() / window_size
df_roll15['refline'] = 0.5
fig15 = df_roll15.plot(kind='line') #, title='15-day rolling accuracy')
fig15.update_layout(xaxis_title="15-day moving average")
fig15.update_xaxes(showgrid=False)
fig15.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')
# fig15.show()


# 30-day
window_size = 30
df_roll30 = df_roll0.replace({-1: 0}).rolling(window=window_size).sum() / window_size
df_roll30['refline'] = 0.5
fig30 = df_roll30.plot(kind='line') #, title='30-day rolling accuracy')
fig30.update_layout(xaxis_title="30-day moving average")
fig30.update_xaxes(showgrid=False)
fig30.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')
# fig30.show()        
        
        
# cumulative hit rate
df_cum = df_roll0.replace({-1: 0}).cumsum()
df_cum_hits = df_cum.div(df_hits['rows'], axis=0)
df_cum_hits['ref_line'] = 0.5
fig_cum = df_cum_hits.plot(kind='line') #, title='Accuracy Trend')
fig_cum.update_xaxes(showgrid=False)
fig_cum.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')
# fig_cum.show()


# cumulative p-value
p1030 = []
p1130 = []
rownum = 1
for i, row in df_cum.iterrows():
    p1030 = p1030 + [stats.binom_test(row['K200f.1030'], n=rownum, p=0.5, alternative='greater')]
    p1130 = p1130 + [stats.binom_test(row['K200f.1130'], n=rownum, p=0.5, alternative='greater')]
    rownum += 1
    
df_pval = pd.DataFrame([p1030, p1130]).T
df_pval.columns = ['K200f.1030', 'K200f.1130']
df_pval.index = df_cum.index
fig_pval = df_pval.plot(kind='line') #, title='Binomial Test p-value')  
fig_pval.update_xaxes(showgrid=False)
fig_pval.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')
# fig_pval.show()


# configure streamlit
st.set_page_config(
    page_title="Wadilt Scorecard Dashboard",
    page_icon="✅",
    layout="wide",
)


# dashboard title
st.title("Wadilt Scorecard Dashboard")


# content
# contianer1 = st.container()
st.sidebar.write('Accuracy & p-value')
st.sidebar.dataframe(df_summary)
st.write('Cumulative Accuracy')
st.plotly_chart(fig_cum, use_container_width=True)
st.write('Binomial Test p-value')
st.plotly_chart(fig_pval, use_container_width=True)     
st.markdown("""---""")
st.write('Moving Averages of Accuracy')
st.plotly_chart(fig5, use_container_width=True)
st.plotly_chart(fig10, use_container_width=True)
st.plotly_chart(fig15, use_container_width=True)
st.plotly_chart(fig30, use_container_width=True)

 




