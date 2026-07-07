#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np


# In[9]:


get_ipython().system('pip install yfinance')


# In[10]:


import yfinance as yf


# In[29]:


9data = yf.download(["AAPL"], start="2024-01-01", end="2024-12-31", auto_adjust=False)
data.head()


# In[37]:


data['R'] = data['Adj Close']['AAPL'].pct_change()
data.head()


# In[41]:


ax = data.R.hist(bins=50)


# In[47]:


data.index = pd.to_datetime(data.index)
monthly_ret = data.R.resample('M').agg(lambda x: (1 + x).prod() -1)
monthly_ret = monthly_ret.to_frame()
monthly_ret.head()


# In[51]:


axs = monthly_ret.R.hist(bins=50)


# In[53]:


monthly_ret.R.mean()


# In[67]:


import matplotlib.pyplot as plt
import scipy.stats as stats


# In[71]:


stats.probplot(monthly_ret.R, dist="norm", plot=plt); #normal tend not to be used


# In[73]:


monthly_ret.R.quantile([0.25, 0.5, 0.75])


# In[75]:


monthly_ret.boxplot('R')


# In[ ]:




