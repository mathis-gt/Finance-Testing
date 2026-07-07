#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import yfinance as yf

tickers = 'SPY QQQ TFT AAPL MSFT GOOG AMZN NFLX NVDA ADBE'

data = yf.download(
    tickers = tickers,
    period = 'max',
    interval = "1d",
    ignore_tz=True,
    auto_adjust=True,
)

data = data['Close']
data["20050101":].head()


# In[3]:


data = data["20050101":]

ticker = "SPY"
close_adj = data[[ticker]].copy()
close_adj.columns = ['close']
close_adj.head()


# In[5]:


close_adj['R'] = close_adj.close.pct_change().fillna(0)
close_adj.head()


# In[7]:


import talib


# In[9]:


close_adj['slow_ma'] = talib.SMA(close_adj.close, 200) #slow/fast moving average
close_adj['fast_ma'] = talib.SMA(close_adj.close, 10)


# In[11]:


close_adj[['close', 'slow_ma']].plot(logy=True, figsize=(10,4))


# In[12]:


close_adj[['slow_ma', 'fast_ma']].plot(logy=True, figsize=(10,4))


# In[15]:


close_adj[['close', 'slow_ma', 'fast_ma']].tail(365).plot()


# In[17]:


close_adj.tail(3)


# In[19]:


close_adj = close_adj[~close_adj.slow_ma.isnull()]


# In[21]:


close_adj = close_adj.assign(
    signal = lambda x: np.where(x.fast_ma > x.slow_ma, 1, 0)
)


# In[23]:


close_adj


# In[25]:


close_adj["2020-05-20":"20200609"]


# In[27]:


close_adj['signal'] = close_adj['signal'].shift(1, fill_value=0)
close_adj['R_strategy'] = close_adj.R * close_adj.signal

close_adj["2020-05-20":"20200609"]


# In[29]:


def get_signal(data, ticker, fast_ma, slow_ma):

    close_adj = data[[ticker]].copy()
    close_adj.columns = ['close']
    
    close_adj['R'] = close_adj.close.pct_change().fillna(0)
    
    close_adj['fast_ma'] = talib.EMA(close_adj.close, fast_ma)
    close_adj['slow_ma'] = talib.EMA(close_adj.close, slow_ma)
    
    close_adj = close_adj[~close_adj.slow_ma.isnull()]
    
    close_adj = close_adj.assign(
    signal = lambda x: np.where(x.fast_ma > x.slow_ma, 1, 0)
    )
    
    close_adj['signal'] = close_adj['signal'].shift(1, fill_value=0)
    close_adj['R_strategy'] = close_adj.R * close_adj.signal
    
    return close_adj


# In[31]:


data_signal = get_signal(data, ticker, fast_ma=10, slow_ma=65)
data_signal.head()


# In[33]:


100 * (1+data_signal[['R', "R_strategy"]]).prod()-1 # total return


# In[35]:


px.line(100 * (1 + data_signal[['R', "R_strategy"]]).cumprod(), title='Total Return')


# In[36]:


def performance (data_signal, ticker, freq= 'M', risk_free_rate=0.02):
    
    rets = data_signal [['R', "R_strategy"]].copy()
    rets.columns = ["Buy & Hold", "Strategy"]
    
    if freq == 'D':
        scale = 252
    elif freq == 'M':
        scale = 12
        rets = rets.resample(freq).agg(lambda x: (1 + x).prod() - 1)
    else:
        return None
        
    ret_cumulative = (1 + rets).cumprod()
    previous_peaks = ret_cumulative.cummax()
    drawdown = (ret_cumulative - previous_peaks) / previous_peaks
    
    annualised_returns = (1 + rets.mean())**scale - 1
    annualied_std_deviation = rets.std()* np.sqrt(scale)
    max_drawdown = drawdown.min() * -1
    
    data_risk_return = pd.DataFrame(
    dict(
    ticker = ticker,
    annualised_returns=annualised_returns,
    annualied_std_deviation=annualied_std_deviation,
    )
    )
    data_risk_return["Max Drawdown"] = drawdown.min() * -1
    
    data_risk_return = data_risk_return.assign(
    sharpe_ratio=lambda x: (x.annualised_returns-risk_free_rate)/x.annualied_std_deviation,
    calmar_ratio=lambda x: (x.annualised_returns)/x['Max Drawdown']
    )
    
    return data_risk_return


# In[37]:


performance(data_signal, ticker, freq='D', risk_free_rate=0.02)


# In[63]:


results = []
for fast_ma in range(5, 50, 1):
    for slow_ma in range(30,200, 1):
        if fast_ma >= slow_ma:
            continue
        else:
            data_signal = get_signal(data, ticker, fast_ma, slow_ma)
            
            perf = performance(data_signal, ticker, freq='D', risk_free_rate=0.02)
            
            perf['fast_ma'] = fast_ma
            perf['slow_ma'] = slow_ma
            results.append(perf.tail(1))


# In[66]:


data_res = pd.concat(results)
data_res


# In[47]:


data_res.sort_values("calmar_ratio", ascending=False).head()


# In[65]:


eval_metric = 'calmar_ratio'

data_mat = data_res.pivot(index='fast_ma', columns='slow_ma', values=eval_metric)

fig = px.imshow(data_mat,
                color_continuous_scale='RdYlGn',
                aspect='auto')

fig.update_layout(
    title=eval_metric,
    xaxis_title='Slow Moving Average',
    yaxis_title='Fast Moving Average',
)
fig


# In[59]:


data_res.sort_values("Max Drawdown", ascending=True).head()


# In[72]:


fast_ma = 10
slow_ma = 70

data_signal = get_signal(data, ticker, fast_ma, slow_ma)
            
perf = performance(data_signal, ticker, freq='D', risk_free_rate=0.02)
perf


# In[70]:


px.line(100 * (1 + data_signal[['R', "R_strategy"]]).cumprod(), title='Total Return')


# In[ ]:




