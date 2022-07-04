# -*- coding: utf-8 -*-
"""
Functions that simulates and loads in data for the thesis

Created on Wed Apr 20 14:31:58 2022

@author: gebruiker
"""
def GenerateNormalData(list_of_tuples, n, correlated_dims, rho):
    """
    Parameters
    ----------
    list_of_tuples : the tuples contain the mean and var of the variables, the length
                     the list determines the amount of variables
    n              : int, amount of observations

    Returns
    -------
    an m by n array of correlated normal data (diagonal covar matrix)

    """
    import numpy as np
  
    array = np.empty((n,len(list_of_tuples)))
    
    for variable in range(len(list_of_tuples)):
        array[:,variable] = np.random.normal(0, 1, n)
        
    amount_of_cols_per_dim = int(len(list_of_tuples) / correlated_dims)
    
    counter = 0
    for i in range(0, correlated_dims):
        for col in range(1, amount_of_cols_per_dim):
            array[:,counter+col] = rho*array[:,counter] + np.sqrt(1-rho**2)* array[:,counter+col]
        counter += amount_of_cols_per_dim
    
    for col in range(len(list_of_tuples)):
        array[:,col] = array[:,col] * list_of_tuples[col][1] + list_of_tuples[col][0]
    
    return array


def GenerateStudentTData(list_of_tuples, n, correlated_dims, rho):
    """
    Parameters
    ----------
    list_of_tuples : the tuples contain the mean, var and degrees of freedom of 
                     the variables, the length of the list determines the 
                     amount of variables
                     
    n              : int, amount of observations

    Returns
    -------
    an m by n array of correlated student t data
    
    t.ppf(x, df, loc, scale)

    """
    from copulae import GaussianCopula
    import numpy as np
    from scipy.stats import t
    array = np.zeros((n,len(list_of_tuples)))
    
    cols_per_dim = int(len(list_of_tuples)/correlated_dims)
    
    if cols_per_dim == 1:
        for col in range(array.shape[1]):
            array[:,col] = np.random.standard_t(df=list_of_tuples[col][2], size=n)
            
    else:
        counter = 0
        for dim in range(correlated_dims):
            cop = GaussianCopula(dim = cols_per_dim)
            cop.params = np.array([rho]*len(cop.params))
            array[:,counter:counter+cols_per_dim] = cop.random(n)
            counter += cols_per_dim
    
    for col in range(array.shape[1]):
        # array[:,col] = t.ppf(array[:,col], df= list_of_tuples[col][2])
        array[:,col] = array[:,col]*list_of_tuples[col][1] + list_of_tuples[col][0]
    
    return array

def GenerateMixOfData(n, rho):
    """
    

    Returns
    -------
    None.

    """
    import numpy as np
    from scipy.stats import bernoulli, t
    from copulae import GumbelCopula
    array = np.zeros((n,12))
    
    
    # normal correlated
    list_of_tuples = [(0,1), (-0.5,0.01), (6,12)]
    array[:,0:3] = GenerateNormalData(list_of_tuples, n, 1, rho)
    
    # student t correlated
    list_of_tuples = [(0,1,8), (-0.5,0.01,4), (6,12,5)]
    array[:,3:6] = GenerateStudentTData(list_of_tuples, n, 1, rho)
    
    # bernoulli correlated
    array[:,6:9] = bernoulli.rvs(0.5, size=(n,3))
    for row in range(5,n):
        corrs = np.corrcoef(array[0:row,6:9], rowvar=False)
        if corrs[0,1] < rho:
            array[row,7] = array[row,6]
        if corrs[0,2] < rho:
            array[row,8] = array[row,6]
        if corrs[1,2] < rho:
            array[row,8] = array[row,7]
    
    # gumbel formula, then transform to student t  (so non linear dependency)
    cop = GumbelCopula(theta=4, dim=3)
    array[:,9:12] = cop.random(n)
    array[:,9] = t.ppf(array[:,9], df=5)
    array[:,10] = t.ppf(array[:,10], df=8)
    array[:,11] = t.ppf(array[:,11], df=6.5)
    
    return array

    
def Yahoo(list_of_ticks, startdate, enddate, retsorclose = 'rets'):
    '''
    Parameters
    ----------
    list_of_ticks : list of strings, tickers
    startdate     : string, format is yyyy-mm-dd
    enddate       : string, format is yyyy-mm-dd
    retsorclose   : string, 'rets' for returns and 'close' for adjusted closing prices
    
    
    Returns
    -------
    dataframe of stock returns or prices based on tickers and date

    '''
    import yfinance as yf
    import pandas as pd
    import numpy as np
    
    dfclose = pd.DataFrame(yf.download(list_of_ticks, start=startdate, end=enddate))['Adj Close']
    dfclose = dfclose.ffill()
    dfclose = dfclose.backfill()
    dfrets  = np.log(dfclose) - np.log(dfclose.shift(1))
    
    if retsorclose == 'rets':
        return dfrets
    else:
        return dfclose

    
def GetData(datatype, correlated_dims=3, rho=0.5):
    """
    Generates a data array based on input

    Parameters
    ----------
    datatype        : string, choose between 'normal', 't', 'mix', 'returns'
    correlated_dims : int, the amount of factors driving the data generation
    rho             : float, the correlation between variables

    Returns
    -------
    array of generated or downloaded data

    """
    
    import numpy as np
    import os
    import pandas as pd
    # dir = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS\data\datasets'
    # dir = r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS\data\datasets'
    dir = os.getcwd()+'\\data\\datasets'


    n = 5000
    filename = datatype+'_rho='+str(rho)+'_dims='+str(correlated_dims)+'.csv'
    
    if filename in os.listdir(dir):
        return np.loadtxt(os.path.join(dir, filename), delimiter=',')
    
    if datatype == 'normal':
        # create check to see if its already generated------------------------------------------------------------------------------------------------
        
        list_of_tuples = [(0,1), (-0.5,0.01), (6,12), (80,10), (-10,6), (100,85),
                          (25, 5), (36, 6), (2, 1), (73, 30), (-10,2.5), (-20, 4)]
        return GenerateNormalData(list_of_tuples, n, correlated_dims, rho)
    
    elif datatype == 't':
        # create check to see if its already generated------------------------------------------------------------------------------------------------
        
        list_of_tuples = [(0,1,4), (-0.5,0.01,4), (6,12,5), (80,10,3), (-10,6,6), (100,85,4.5),
                          (25, 5,5), (36, 6, 6), (2, 1, 8), (73, 30, 5), (-10,2.5,10), (-20, 4, 4.44)]
        return GenerateStudentTData(list_of_tuples, n, correlated_dims, rho)
    
    elif datatype == 'returns':
        # create check to see if its already there   ------------------------------------------------------------------------------------------------

        X = pd.read_csv(os.getcwd()+'\\data\\datasets\\real_sets\\masterset_returns.csv')#.drop(0, axis=0)
        X['Name'] = pd.to_datetime(X['Name'], infer_datetime_format=True)
        X = X[X['Name'] > pd.to_datetime('01/01/2010')]
        X = X.ffill()
        X = X.backfill()
        X = np.array(X.iloc[:,1:])
        X = X.astype(float) # now we have prices, so we need to figure out weights
        weights = np.empty((X.shape[0], X.shape[1]))
        for row in range(X.shape[0]):
            weights[row,:] = X[row,:]/sum(X[row,:])
        X = np.log(X[1:,:]) - np.log(X[:-1,:])
        return (X, weights[1:,:])
    
    elif datatype == 'mix':
        # create check to see if its already generated------------------------------------------------------------------------------------------------
        
        return GenerateMixOfData(n,rho)
    
    elif datatype == 'interestrates':
        print('This is gonna be a feature, but its not done yet!')
    else:
        print('datatype not recognized, please consult docstring for information on valid data types')
        

def GenerateAllDataSets(delete_existing = False):
    """
    Function that writes all simulated datasets needed for VAE performance analysis

    Parameters
    ----------
    delete_existing : bool, optional
        If true, this function will delete the existing simulating datafiles and generate new ones. The default is False.

    Returns
    -------
    None.

    """   
    if delete_existing:
        # delete all datasets that are already there
        import numpy as np
        import os
        # dir = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS\data\datasets'
        # dir = r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS\data\datasets'
        dir = os.getcwd()+'\\data\\datasets'
        for file in os.listdir(dir):
            os.remove(os.path.join(dir,file))
        
        for rho in [0.25, 0.5, 0.75]:
            for correlated_dims in [2,3,4,6,12]:
                X_normal = GetData('normal', correlated_dims, rho)
                np.savetxt(os.path.join(dir, 'normal_rho='+str(rho)+'_dims='+str(correlated_dims)+'.csv'), X_normal, delimiter=',')
                X_t      = GetData('t', correlated_dims, rho)
                np.savetxt(os.path.join(dir, 't_rho='+str(rho)+'_dims='+str(correlated_dims)+'.csv'), X_t, delimiter=',')
            X_mix = GetData('mix', correlated_dims, rho)
            np.savetxt(os.path.join(dir, 'mix_rho='+str(rho)+'.csv'), X_mix, delimiter=',')

    return


    
    
    
    
    
    
    
    