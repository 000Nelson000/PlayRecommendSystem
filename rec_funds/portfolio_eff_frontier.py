# -*- coding: utf-8 -*-
"""
依據Markowitz資產管理理論，建立最佳效率前緣(efficient frontier)資產配置

算法參考自: https://blog.quantopian.com/markowitz-portfolio-optimization-2/

Created on Fri Aug 10 16:30:51 2018

@author: 116952
"""

import pandas as pd 
import numpy as np 
import pypyodbc 
#import seaborn as sns
import matplotlib.pyplot as plt

import cvxopt as opt
from cvxopt import blas, solvers

# Turn off progress printing 
solvers.options['show_progress'] = False
#%% 

def gen_weight(n):
    '''隨機產生權重配置,總和=1'''
    k = np.random.rand(n)
    return k/sum(k)

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(gen_weight(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def get_mean_std(returns):
    std = returns.std()
    mean = returns.mean()
    return mean,std

def gen_portfolio(returns):
    '''
    產生資產組合平均值(mean)與標準差(std)
    '''
    n = len(returns)
    n =1 
    p = np.asmatrix(returns.mean(axis=0)) 
    w = np.asmatrix(gen_weight(returns.shape[1]))
    C = np.asmatrix(returns.cov())    
    mu = n*w*p.T
    sigma = np.sqrt(n*w*C*w.T) ## 52週
    return mu,sigma


def optimal_portfolio(r):

    n =len(r)
#    num = r.shape[1]
    num=1
    ret = np.asmatrix(r)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(ret)) ## covariance , diagonal is the variance of each stock
    pbar = opt.matrix(np.mean(ret, axis=1)) #
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    ret = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(ret, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO 
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), num*np.array(ret), np.sqrt(1.0*num)*np.array(risks), portfolios


def plot_eff_weight(portfolios,risks):
    '''劃出最佳效率前緣下的不同資產權重,另以虛線表示中等風險程度的資產配置 '''
    assert risks.shape[0] == len(portfolios), 'len should be same'
    w_array = np.array([np.array(x).ravel() for x in portfolios])
    eff_w_df = pd.DataFrame(w_array).add_prefix('w')
#    eff_w_df['risk'] = risks
    eff_cum_df = eff_w_df.cumsum(axis=1)
    eff_cum_df['risk'] = risks
    eff_cum_df.plot(x='risk')
    
    plt.xlabel('standard deviation')
    plt.ylabel('allocation')
    plt.title('Optimal allocations')
    arg_mean_min = np.abs(eff_cum_df.risk - eff_cum_df.risk.mean()).idxmin()
    risk_mean = eff_cum_df.risk[arg_mean_min]    
    
    plt.vlines(risk_mean,0,1,linestyles='--')
    
    return eff_w_df.iloc[arg_mean_min,:], risk_mean
#%%
if __name__ == '__main__':
    
    '''讀取基金歷史資料/計算基金相關性'''
    
    
    con = pypyodbc.connect(
        "DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=external"
    )
    read_sql = ''' select 基金類型,
                    基金代碼,
                    淨值,
                    convert(varchar(8),更新時間,112) 更新時間 
                    from dbo.MMA基金基本資料_每週更新v2 '''
    
    df = pd.read_sql(read_sql,con)
    df['更新時間'] = pd.to_datetime(df['更新時間'])
    price_df = df.set_index('更新時間')[['基金代碼','淨值']]
    price_df_ =  pd.pivot_table(price_df, values='淨值',index=price_df.index,columns='基金代碼')
    filter_idx = price_df_.index > pd.datetime(2017,7,20)
    price_df = price_df_[filter_idx].fillna(method='ffill')
    corr_funds = price_df.corr()

#    sns.heatmap(corr_funds)
    #%% 
    '''任選基金(測) 計算 risk(std) - return 分布'''
    test_funds = ['83F','MB2','J14','V09','T34','Y38','J79','MU7','213','L84'] ## 
#    test_funds = ['T34','T38','UI4','86H'] ## neg correlate
#    test_funds = ['T38','Y38','MU7','J84','MC5']
    price_test = price_df.loc[:,test_funds]    
    returns = (price_test.pct_change().dropna())
    

#    returns *= 100
#    returns.cumsum().plot()
    
    
    
    n_portfolios = 200
    means, stds = np.column_stack([
        gen_portfolio(returns) 
        for _ in range(n_portfolios)
        ])
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.title('mean/std returns of randomly gen portfolios')
#    #%%
##    '''亂數組合return(測試用) '''
#    ## NUMBER OF ASSETS
#    n_assets = 3
#    
#    ## NUMBER OF OBSERVATIONS
#    n_obs = 1000
#    
#    return_vec = np.random.randn(n_assets, n_obs)
#    n_portfolios = 500
#    means, stds = np.column_stack([
#        random_portfolio(return_vec) 
#        for _ in range(n_portfolios)
#        ])
#        
#    plt.plot(stds, means, 'o', markersize=5)
#    plt.xlabel('std')
#    plt.ylabel('mean')
#    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    
    #%%% 
    '''最佳化效率前緣 '''
    
#    weights, ret, risks,portfolios = optimal_portfolio(return_vec) ## 隨機
    weights, ret, risks,portfolios = optimal_portfolio(returns.T) ## 
    plt.plot(stds*np.sqrt(52), means*52, 'o')
    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(risks*np.sqrt(52), ret*52, 'y-o')
    
    

#    plt.plot(np.sqrt(52)*stds, 52*means,'o')
#    plt.plot(risks*np.sqrt(52), ret*52,'g-o')
    #%% backtest
    print(weights.ravel())
#    opt_risk_rand_ret = pd.DataFrame({'return':return_vec.mean(),
#                                      'w':weights.ravel(),
#                                      'risk':np.std(return_vec,axis=1)})
#    returns.cumsum().iloc[-1,:]
#    returns.std()
    
    num =52
    
    '''最佳解:績效最好 (通常只選到過往績效最佳的基金/股票), 反之風險最小亦同'''
    
    opt_risk_ret = pd.DataFrame({'return':returns.mean()*num,
                                 'w':weights.ravel(),
                                 'risk':returns.std()*np.sqrt(num)})
    ####若追求中等風險/投資分散 ######
    weights_mean,risk_mean = plot_eff_weight(portfolios,risks*np.sqrt(52))
    weights_mean.index = test_funds
    mean_risk_ret = pd.DataFrame({'return': returns.mean()*(52),
                                 'w': weights_mean,
                                 'risk': returns.std()*np.sqrt(52)})
    
    