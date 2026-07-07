#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import yfinance as yf


# In[2]:


df = pd.read_csv("ishares.csv")
df.head


# In[5]:


stocks = ['SPY', 'MSFT', 'META', 'AMZN', 'MS']
ishares = ['IGUS.L', 'EEDG.L', 'IGTM.L', 'LQGH.L', 'SAEM.L', 'CBUG.L', 'SEGP.L', 'EMHG.L', 'EWSP.L', 'CUKX.L', 'UESD.L', 'EUXS.L', 'IGLT.L', 'SUOG.L', 'IEUX.L', 'EEJG.L', 'SGLN.L', 'IGLS.L', 'IBTG.L', 'GSPX.L', 'SEML.L', 'IWDP.L', 'EHYG.L', 'CPJ1.L', 'TI5G.L', 'SLXX.L', 'DHYG.L', 'INXG.L']


s_data = yf.download(
    tickers = stocks,
    start='2012-01-01', 
    end='2025-11-01', 
    interval='1wk',
    ignore_tz=True,
    auto_adjust=True
    )
 
i_data = yf.download(
    tickers = ishares,
    start='2012-01-01', 
    end='2025-11-01', 
    interval='1wk',
    ignore_tz=True,
    auto_adjust=True
    )


# In[7]:


s_data = s_data.pct_change()
s_data = s_data.fillna(0)
s_data.index = pd.to_datetime(s_data.index)
s_monthly_ret = s_data.resample('ME').agg(lambda x: (1 + x).prod() -1)
s_ret_cumulative = (1 + s_monthly_ret).cumprod()


# In[9]:


i_data = i_data.pct_change()
i_data = i_data.fillna(0)
i_data.index = pd.to_datetime(i_data.index)
i_monthly_ret = i_data.resample('ME').agg(lambda x: (1 + x).prod() -1)
i_ret_cumulative = (1 + i_monthly_ret).cumprod()


# In[10]:


px.line(
    s_ret_cumulative['Close']
        .filter(items=[t for t in stocks if t in s_ret_cumulative['Close'].columns], axis=1)
        .pipe(lambda df: df / df.iloc[0]),
    log_y=True
)


# In[13]:


px.line(
    i_ret_cumulative['Close']
        .filter(items=[t for t in ishares if t in i_ret_cumulative['Close'].columns], axis=1)
        .pipe(lambda df: df / df.iloc[0]),
    log_y=True
)


# In[ ]:


# find a way to group them as I grouped them
# graph these groups
# do an individual risk assessment for stocks then group risk assessment for ishares


# In[15]:


s_annualised_returns = (1 + s_monthly_ret.mean())**12 -1
s_annualised_std_deviation = s_monthly_ret.std() * np.sqrt(12)
s_annualised_std_deviation


# In[17]:


s_data_risk_return = pd.DataFrame(
    dict(s_annualised_returns=s_annualised_returns,
        s_annualised_std_deviation=s_annualised_std_deviation,
        )
)

risk_free_rate = 0.02

s_data_risk_return['sharpe_ratio'] = (s_data_risk_return['s_annualised_returns'] - risk_free_rate) / s_data_risk_return['s_annualised_std_deviation']
s_data_risk_return.sort_values("sharpe_ratio", ascending=False)


# In[19]:


s_previous_peaks = s_ret_cumulative.cummax()

s_drawdown = (s_ret_cumulative - s_previous_peaks) / s_previous_peaks
px.line(s_drawdown['Close'])

s_data_risk_return["Max Drawdown"] = s_drawdown.min() * -1
s_data_risk_return = s_data_risk_return.reset_index().rename(columns={"index":"ticker"})
s_data_risk_return = s_data_risk_return.drop_duplicates(subset='Ticker')


# In[25]:


fig = px.scatter(s_data_risk_return,
                 x="s_annualised_returns",
                 y="s_annualised_std_deviation",
                 text="Ticker")

fig.update_traces(textposition='top center')

fig.update_layout(
    xaxis=dict(title='Annualised Returns', tickformat=".0%"),
    yaxis=dict(title='Standard Deviation', tickformat=".0%"),
    title_text='Annualised Returns & Risk'
)

fig.show()


# In[ ]:




