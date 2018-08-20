# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:30:51 2018

@author: 116952
"""

import pandas as pd 
import numpy as np 
import pypyodbc 
import seaborn as sns
#%%
if __name__ == '__main__':
    
    con = pypyodbc.connect(
        "DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=external"
    )
    read_sql = ''' select 基金類型,基金代碼,淨值,淨值日期 from dbo.MMA基金基本資料_每週更新v2'''
    
    df = pd.read_sql(read_sql,con)
    
    price_df = df.set_index('淨值日期')[['基金代碼','淨值']]
    price_df_ =  pd.pivot_table(price_df, values='淨值',index=price_df.index,columns='基金代碼')
    filter_idx = price_df_.index > pd.datetime(2017,7,20)
    price_df = price_df_[filter_idx].fillna(method='ffill')
    corr_funds = price_df.corr()

#    sns.heatmap(corr_funds)
    #%% 