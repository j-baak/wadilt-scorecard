# import packages
import os, boto3, datetime
import pandas as pd
import numpy as np
import plotly.express as px
pd.options.plotting.backend = "plotly"
import matplotlib.pyplot as plt
from sns_cm import sns_confusion_matrix

import streamlit as st  # 🎈 data web app development
from scipy import stats

# configure streamlit
st.set_page_config(
    page_title="Wadilt Scorecard Dashboard",
    page_icon="✅",
    layout="wide"
    # layout="centered"
)

# dashboard title
# st.title("Wadilt Scorecard Dashboard")
st.markdown("<h1 style='text-align: left; color: #1E90FF;'>Wadilt Scorecard Dashboard</h1>", unsafe_allow_html=True)

if "load_state" not in st.session_state:
    st.session_state.load_state = False

if st.button('Load Data') or st.session_state.load_state:
    st.session_state.load_state = True

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


    # onvert to classification (up, down) matrix
    df_pred = df_pred.apply(np.sign).replace({1: 'up', -1: 'down'})
    df_pred.columns = ['prediction']
    df_ground = df_ground.apply(np.sign).replace({1: 'up', -1: 'down'})
    df_ground.columns = ['K200f.1030 (at timepoint)', 'K200f.1130 (mean after open)']
    df_class = pd.concat([df_pred, df_ground], axis=1)    
    df_class.index = df_class.index.strftime('%Y-%m-%d')

    # confusion matrix
    from sklearn.metrics import confusion_matrix #ConfusionMatrixDisplay    
    cm1030 = confusion_matrix(
            y_true=df_class['K200f.1030 (at timepoint)'],
            y_pred=df_class['prediction'],
            labels=['up', 'down']  # rearange labels
    )

    cm1130 = confusion_matrix(
            y_true=df_class['K200f.1130 (mean after open)'],
            y_pred=df_pred['prediction'],
            labels=['up', 'down']  # rearange labels
    ) 
    
    vmin = np.concatenate((cm1130.flatten(), cm1030.flatten())).min()
    vmax = np.concatenate((cm1130.flatten(), cm1030.flatten())).max()

    sns_confusion_matrix(cm1030, cbar_range=(vmin, vmax), categories=['up', 'down'], title='K200f.1030')
    plt.savefig('cm_1030.png', bbox_inches='tight')
    sns_confusion_matrix(cm1130, cbar_range=(vmin, vmax), categories=['up', 'down'], title='K200f.1130')
    plt.savefig('cm_1130.png', bbox_inches='tight')


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


    # # configure streamlit
    # st.set_page_config(
    #     page_title="Wadilt Scorecard Dashboard",
    #     page_icon="✅",
    #     layout="wide"
    #     # layout="centered"
    # )


    # # dashboard title
    # st.title("Wadilt Scorecard Dashboard")

    # # content
    # # contianer1 = st.container()

    # st.sidebar.write('accuracy & p-value')
    # st.sidebar.dataframe(df_summary)
    # st.sidebar.write('confusion matrix - K200f.1030')
    # st.sidebar.write(cm1030)
    # st.sidebar.write('confusion matrix - K200f.1130')
    # st.sidebar.write(cm1130)
    # st.sidebar.markdown("""---""")

    st.markdown("""---""")
    
    st.subheader('accuracy & p-value')
    st.dataframe(df_summary)
    st.subheader('confusion matrix')
    col_1, col_2 = st.columns([1, 1])
    with col_1: 
        # st.write('confusion matrix - K200f.1030')        
        # st.pyplot(cm_1030, bbox_inches='tight')
        st.image('cm_1030.png')
        
    with col_2:
        # st.write('confusion matrix - K200f.1130')
        # st.pyplot(cm_1130, bbox_inches='tight')
        st.image('cm_1130.png')

    json_loc = 'https://bucket-4-clients.s3.ap-northeast-2.amazonaws.com/predictions_K200f/prediction_K200f_yyyymmdd.json'
    png_loc = 'https://bucket-4-clients.s3.ap-northeast-2.amazonaws.com/predictions_K200f/prediction_K200f_yyyymmdd.png'

    with st.expander('view data'):
        st.dataframe(df_class)
        st.write('Historical predictions data on AWS: ' + json_loc)
        st.write('Historical prediction plots on AWS: ' + png_loc)        
        st.caption('(replace "yyyymmdd" with particular date, e.g. "20220223")')
        st.markdown("""---""")
        st.write('first prediction: ' + df_pred.index[0].strftime('%Y-%m-%d'))
        st.write('last prediction: ' + df_pred.index[-1].strftime('%Y-%m-%d'))
        st.write('number of valid predictions: ' + str(len(df_pred)) + ' (prediction is invalid if models disagree)')

    st.subheader('Cumulative Accuracy')
    st.plotly_chart(fig_cum, use_container_width=True)
    st.subheader('Binomial Test p-value')
    st.plotly_chart(fig_pval, use_container_width=True)     
    st.markdown("""---""")
    st.subheader('Moving Averages of Accuracy')
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig10, use_container_width=True)
    st.plotly_chart(fig15, use_container_width=True)
    st.plotly_chart(fig30, use_container_width=True)

# ========== Using plotly subplots ============
# ========== can't avoid duplicate lengeds ===
# from plotly.subplots import make_subplots
# fig1 = make_subplots(rows=2, cols=1, subplot_titles=("Accuracy", "Binomial Test p-value"))
# fig1.add_trace(fig_cum.data[0], row=1, col=1)
# fig1.add_trace(fig_cum.data[1], row=1, col=1)
# fig1.add_trace(fig_cum.data[2], row=1, col=1)

# fig1.add_trace(fig_pval.data[0], row=2, col=1)
# fig1.add_trace(fig_pval.data[1], row=2, col=1)
# fig1.update_layout(height=800, title_text="Model Performance")
# fig1.update_xaxes(showgrid=False)
# fig1.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')
 
# fig2 = make_subplots(rows=4, cols=1, subplot_titles=("5-day span", "10-day span", "15-day span", "30-day span"))
# fig2.add_trace(fig5.data[0], row=1, col=1)
# fig2.add_trace(fig5.data[1], row=1, col=1)
# fig2.add_trace(fig5.data[2], row=1, col=1)

# fig2.add_trace(fig10.data[0], row=2, col=1)
# fig2.add_trace(fig10.data[1], row=2, col=1)
# fig2.add_trace(fig10.data[2], row=2, col=1)

# fig2.add_trace(fig15.data[0], row=3, col=1)
# fig2.add_trace(fig15.data[1], row=3, col=1)
# fig2.add_trace(fig15.data[2], row=3, col=1)

# fig2.add_trace(fig30.data[0], row=4, col=1)
# fig2.add_trace(fig30.data[1], row=4, col=1)
# fig2.add_trace(fig30.data[2], row=4, col=1)

# fig2.update_layout(height=1000, title_text="Moving Averages of Accuracy")
# fig2.update_xaxes(showgrid=False)
# fig2.update_yaxes(range=[0, 1], showgrid=True, gridwidth=0.1, griddash='dot')

# st.sidebar.write('Accuracy & p-value')
# st.sidebar.dataframe(df_summary)

# st.plotly_chart(fig1, use_container_width=True)
# st.plotly_chart(fig2, use_container_width=True)


# ================ Using pandas-bokeh =============
# ==== groidplot doesn't work (RuntimeError: Models must be owned by only a single document,...)
# ==== st.columns doesn't work (plot widths gets screwed up)
# import pandas_bokeh
# from bokeh.layouts import gridplot
# from bokeh.plotting import output_notebook, figure, show
# pd.set_option('plotting.backend', 'pandas_bokeh')
# pandas_bokeh.output_notebook()

# f1 = df_cum_hits.plot()
# f2 = df_pval.plot()

# LO = pandas_bokeh.plot_grid([[f1], [f2]])
                        
# LO = pandas_bokeh.plot_grid([[f1, f2]])
# LO = gridplot([[f1, f2]], plot_width=250, plot_height=250)
# st.bokeh_chart(LO)
# st.write(LO)

# col1, col2 = st.columns(2)
# col1.bokeh_chart(f1)
# col2.bokeh_chart(f2)

# ========================= Using Bokeh natively =============
# ===== this example works
# x = list(range(11))
# y0 = x
# y1 = [10 - i for i in x]
# y2 = [abs(i - 5) for i in x]

# s1 = figure(background_fill_color="#fafafa")
# s1.circle(x, y0, size=12, alpha=0.8, color="#53777a")

# s2 = figure(background_fill_color="#fafafa")
# s2.triangle(x, y1, size=12, alpha=0.8, color="#c02942")

# s3 = figure(background_fill_color="#fafafa")
# s3.square(x, y2, size=12, alpha=0.8, color="#d95b43")

# grid = gridplot([[s1, s2, s3]], plot_width=250, plot_height=250)

# st.bokeh_chart(grid)


