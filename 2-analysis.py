#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf


# In[4]:


stocks = 'SPY QQQ TLT AAPL MSFT GOOG AMZN NFLX NVDA MA ADBE'

data = yf.download(
    tickers = stocks,
    start='2024-01-01', 
    end='2025-10-26', 
    interval='1wk',
    ignore_tz=True,
    auto_adjust=True
    )


# In[7]:


data.head()


# In[9]:


data = data.pct_change()
data.head()


# In[47]:


data = data.fillna(0)
data.head()


# - Spy: S&P 500
# - QQQ: Nasdaq-100
# - TLT: Long term bond US treasury ETF

# In[50]:


data.index = pd.to_datetime(data.index)
monthly_ret = data.resample('ME').agg(lambda x: (1 + x).prod() -1)
monthly_ret.head()


# In[52]:


monthly_ret.corr()


# In[54]:


ret_cumulative = (1 + monthly_ret).cumprod()
ret_cumulative


# In[56]:


import plotly.express as px


# In[60]:


px.line(
    ret_cumulative['Close'].filter(items=['SPY', 'NVDA', 'TLT'], axis=1), 
    log_y=True
)


# In[62]:


px.line(
    (ret_cumulative['Close'].filter(items=['SPY', 'NVDA', 'TLT'], axis=1) /
     ret_cumulative['Close'].filter(items=['SPY', 'NVDA', 'TLT'], axis=1).iloc[0]),
    log_y=True
)


# In[76]:


stocks = ['SPY', 'QQQ', 'TLT', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NFLX', 'NVDA', 'MA', 'ADBE']

px.line(
    ret_cumulative['Close']
        .filter(items=[t for t in stocks if t in ret_cumulative['Close'].columns], axis=1)
        .pipe(lambda df: df / df.iloc[0]),
    log_y=True
)


# In[86]:


annualised_returns = (1 + monthly_ret.mean())**12 -1
annualised_returns


# In[88]:


annualised_std_deviation = monthly_ret.std() * np.sqrt(12)
annualised_std_deviation


# In[90]:


data_risk_return = pd.DataFrame(
    dict(annualised_returns=annualised_returns,
        annualised_std_deviation=annualised_std_deviation,
        )
)

risk_free_rate = 0.02

data_risk_return['sharpe_ratio'] = (data_risk_return['annualised_returns'] - risk_free_rate) / data_risk_return['annualised_std_deviation']

data_risk_return.sort_values("sharpe_ratio", ascending=False)


# In[92]:


previous_peaks = ret_cumulative.cummax()

drawdown = (ret_cumulative - previous_peaks) / previous_peaks

drawdown


# In[104]:


px.line(drawdown['Close'][['SPY', 'NVDA', 'TLT']]) #tells you about them drops


# In[108]:


data_risk_return["Max Drawdown"] = drawdown.min() * -1
data_risk_return # max drawdown says largest percentage drop


# In[110]:


data_risk_return = data_risk_return.reset_index().rename(columns={"index":"ticker"})
data_risk_return


# In[120]:


data_risk_return = data_risk_return.drop_duplicates(subset='Ticker')

fig = px.scatter(data_risk_return,
                 x="annualised_returns",
                 y="annualised_std_deviation",
                 text="Ticker")

fig.update_traces(textposition='top center')

fig.update_layout(
    xaxis=dict(title='Annualised Returns', tickformat=".0%"),
    yaxis=dict(tickformat=".0%"),
    title_text='Annualised Returns & Risk'
)

fig.show()


# In[ ]:




