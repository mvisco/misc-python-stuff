import datetime

import pandas as pd
import pandas.io.data as pd_io
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import threading

mpl.rc('figure', figsize=(8, 7))

#aapl = pd.io.data.get_data_yahoo('AAPL', start=datetime.datetime(2006, 10, 1),end=datetime.datetime(2012, 1, 1))
aapl=pd_io.get_data_yahoo('AAPL', start=datetime.datetime(2014, 7, 1),end=datetime.datetime(2014, 12, 31))
googl=pd_io.get_data_yahoo('GOOGL', start=datetime.datetime(2014, 7, 1),end=datetime.datetime(2014, 12, 31))
fb = pd_io.get_data_yahoo('FB', start=datetime.datetime(2014, 7, 1),end=datetime.datetime(2014, 12, 31))
qqq=pd_io.get_data_yahoo('QQQ', start=datetime.datetime(2014, 7, 1),end=datetime.datetime(2014, 12, 31))
msft=pd_io.get_data_yahoo('AMZN', start=datetime.datetime(2015, 1, 1),end=datetime.datetime(2015, 5, 31))

#aapl_adj_close = (aapl['High']-aapl['Low'])/((aapl['High']+aapl['Low'])/2.0)
#googl_adj_close=(googl['High']-googl['Low'])/((googl['High']+googl['Low'])/2.0)

aapl_adj_close = (aapl['High']-aapl['Low'])
googl_adj_close=(googl['High']-googl['Low'])
fb_adj_close = (fb['High']-fb['Low'])
qqq_adj_close=qqq['High']-qqq['Low']
msft_adj_close=msft['High']-msft['Low']


fb_adj_close.plot(label='FB')
plt.legend()
plt.show()



googl_adj_close.plot(label='GOOGL')
plt.legend()
plt.show()

aapl_adj_close.plot(label='AAPL')
plt.legend()
plt.show()

qqq_adj_close.plot(label='QQQ')
plt.legend()
plt.show()

msft_adj_close.plot(label='MSFT')
plt.legend()
plt.show()


