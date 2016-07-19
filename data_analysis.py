#!/usr/bin/python2.7
import MySQLdb
import numpy as np
import pandas as pd
from numpy import mean
from scipy import std
from numpy import cumsum, log, polyfit, sqrt, std, subtract, core
from numpy.random import randn
import sys
import sklearn
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
import statsmodels.tsa.stattools as stat
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading as thread
import  scipy.stats as stats 

def data_analysis():

 
    #print datetime.date.today()
    #print datetime.datetime.today()
    db = MySQLdb.connect(host="localhost", # your host, usually localhost
                     user="root", # your username
                      passwd="password", # your password
                      db="traderlight2014") # name of the data base
    cur = db.cursor()
    
    #start_index = 28579056
    #end_index = 28629056
    #end_index= 29187004
    
   # start_index = 32809600
    #end_index=32893456
   # end_index=33427224
    
   # start_index=32809600
    ##end_index=32859600
    #end_index=33427224
    
    #start_index=33427224
    #end_index=34048415
    #start_index=33427224
    #end_index=34678954
    
    start_index= 7018128                #6888888     #7018128
    end_index= 7149078                 #7018128             #7149078
    
    #start_index=5076774
    #end_index=5205870
    
    nr=range(0,391)   
    ts=pd.DataFrame(index=nr, columns=['bid','ask','volume'])
    list_symbol=[]
    
    k=(end_index- start_index)/390
    
    
    for n in range(0,390):       
        #exec_string="select * from traderlight2014.levelonequote where id=" + str(n) + ";"
        #exec_string1="select * from traderlight2014.levelonequote where (id=" + str(n) + " and symbol='GOOGL')" + ";"
        #print exec_string1
        
        start_n=start_index+n*k
        end_n=start_n+40
        #cur.execute('SELECT * FROM traderlight2014.levelonequote WHERE id=%s and symbol=%s;',(str(n),'INTC'))
        cur.execute('SELECT * FROM traderlight2014.levelonequote WHERE id between %s and %s and symbol=%s;',(str(start_n),
                                                                                                             str(end_n),'QQQ'))
        #cur.execute(exec_string1)
        # print all the first cell of all the rows
        a = cur.fetchall()
        if a :
            b=a[0]
            pos=position(b[2])
            #print pos
            if pos == -1:
                continue
            #print ts['bid'][pos]
            if pd.isnull(ts['bid'][pos]):
                print b
                ts['bid'][pos]=float(b[5])
                ts['ask'][pos]=float(b[6])
                ts['volume'][pos]=b[9]
                list_symbol.append(float(b[5]))
     
    ts['bid'][389]=ts['bid'][388]
    ts['bid'][390]=ts['bid'][388]
    #print ts['bid']
    print ts
    c=np.asarray(ts['bid'])
    d=np.asarray(list_symbol)
    diff_list = []
    std_dev_array=[]
    list_for_mean=[]
    mean_list=[]
    for a in d:
        diff_list.append(a - d.mean())
        list_for_mean.append(a)
        array_for_mean=np.asarray(list_for_mean)
        mean_list.append(array_for_mean.mean())
        diff_array=np.asarray(diff_list)
        std_dev_array.append(diff_array.std())
    print d
    print 'mean is ', d.mean()
    print 'std_dev is ',d.std()
   # models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
   # for m in models:
   #     fit_model(m[0], m[1], X_train, y_train, X_test, pred)      
    max_lag=d.size-1
    p=stat.adfuller(d, 1, "nc", store="true")
    result=p[3]
    print result.resols.summary()
    print p
    b=[]
    for a in d:
        b.append(np.core.umath.log10(a))
    h=hurst(b,max_lag)
    print 'h is ', h
    #d.plot(label='GOOGL') 
    
    #np.corrcoef(x, y, rowvar, bias, ddof)
    
    plt.subplot(3, 1, 1)
    plt.plot(d)
    plt.plot(mean_list)
    plt.ylabel('Prices') 
    plt.xlabel('Minutes')
    #plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(std_dev_array) 
    plt.ylabel('std_dev') 
    plt.xlabel('Minutes')
    #plt.legend()
    plt.subplot(3, 1, 3)
    fit = stats.norm.pdf(d, np.mean(d), np.std(d))  #this is a fitting indeed
    plt.plot(d,fit,'-o')
    plt.hist(d,bins=40,normed=True,histtype='step')
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
      
    plt.show()


def position(a):
    
    minute=a.minute
    sec= a.second
    hour=a.hour
    
    print a
    print minute
    print sec
    print hour
    for i in range(0,60):
        if minute==i:
            if hour == 7:
                return (i-30) 
            if hour == 8:
                return(i+30)
            if hour == 9:
                return(i+90)
            if hour == 10:
                return(i+150)
            if hour == 11:
                return(i+210)
            if hour == 12:
                return(i+270)
            if hour == 13:
                return(i+329)
            if hour >= 14:
                return 390  
            
def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)

    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print "%s: %.3f" % (name, hit_rate)     
    mpl.rc('figure', figsize=(8, 7))

def hurst(p,n):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, n)
    tau = []; lagvec = []
    for lag in lags:  
        #  produce price difference with lag  
        pp = subtract(p[lag:],p[:-lag])  
        #  Write the different lags into a vector  
        lagvec.append(lag)  
        #  Calculate the variance of the differnce vector  
        tau.append(sqrt(std(pp))) 

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0



        
if __name__ == "__main__":
    data_analysis()    
    