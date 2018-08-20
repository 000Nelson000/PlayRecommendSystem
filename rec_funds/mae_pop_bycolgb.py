# -*- coding: utf-8 -*-

'''利用級距切出全行不同族群之[投資老手]熱銷基金,並藉此給出[舊戶]之推薦清單
----------------------------------------------------------------------
* 
    - 資料來源 : project2018.dbo.v_基金推薦_投資戶標籤
* 目的
    - 計算
        - mae : 評估變數差異性
        - 新手recall: 以各種群組熱銷組合評估新手戶之準確度
---------------------------------------------------------------------
Created on Wed Aug 15 11:08:00 2018

@author: 116952
'''
#%%
import pandas as pd
import numpy as np
import pypyodbc
from itertools import product
from collections import defaultdict
from sqlalchemy import create_engine
import sqlalchemy


def getTopSellsFundsOrder(df):
    '''取得基金熱銷排名,與對應的銷售憑證數(賣幾張)
    '''
    top_sells_funds = df.groupby('基金代碼')['憑證'].count().reset_index().rename(
        columns={'憑證': '憑證數'}).sort_values(by='憑證數', ascending=False)
    top_sells_funds['熱銷排名'] = range(1, top_sells_funds.shape[0] + 1)
    return top_sells_funds


def getTopSellsFundsOrderByCols(df, gb):
    '''在群組條件內取得基金熱銷排名,與對應憑證數'''
    assert isinstance(gb, list), 'gb should be list'
    gb_cols = gb + ['基金代碼']
    sort_cols = gb + ['憑證數']
    df_r = df.groupby(gb_cols)['憑證'].count().to_frame().rename(
        columns={'憑證': '憑證數'}).reset_index().sort_values(by=sort_cols, ascending=False)
    df_r['群組排名'] = df_r.groupby(gb).cumcount() + 1
    return df_r


def cal_mae_df(df, gb, topn=10, weight=False):
    '''計算mae
    '''
    pop_df = getTopSellsFundsOrder(df)
    pop_df2 = pop_df[['基金代碼', '熱銷排名']]
    gb_df = getTopSellsFundsOrderByCols(df, gb)
    mae_df = pd.merge(gb_df, pop_df2, how='left', on='基金代碼')
    mae_df['排名差異'] = abs(mae_df.群組排名 - mae_df.熱銷排名)
    mae_df = mae_df.groupby(gb).head(topn)
    # mae_df_tmp = mae_df.copy()

    if not weight:
        mae_group = mae_df.groupby(gb).sum() / topn
        mae_group = mae_group.rename(columns={'排名差異': 'mae'})['mae']
        mae = mae_group.mean()
    else:
        mae_df['w_by_gb'] = mae_df['憑證數'] / \
            mae_df.groupby(gb)['憑證數'].transform('sum')
        mae_df['diff_times_w'] = mae_df['排名差異'] * mae_df['w_by_gb']
        mae_g = mae_df.groupby(gb)['diff_times_w'].sum()
        gb_sum = mae_df.groupby(gb)['憑證數'].sum()
        gb_w = gb_sum / gb_sum.sum()
        mae = (mae_g * gb_w).sum()
    return mae_df, mae


def cut_df(df, col, cut_no=5,percentile=True):
    '''計算等寬級距(cut),或等比例級距(qcut)
    ====
    df    : (dataframe)
    cut_no: (int) 
    percentile: (bool)
    '''
    col_range = col + '_級距'
    df.drop(col_range,axis=1,inplace=True) if col_range in df.columns else df    
    tmp_df = df.copy()
    tmp = tmp_df[['身分證字號', col]].drop_duplicates()
    if isinstance(cut_no,list):
        lab = range(len(cut_no)-1)
    if isinstance(cut_no,int):
        lab = range(cut_no)
        
    if percentile:
        tmp[col_range],bins = pd.qcut(tmp[col], cut_no,retbins=True,labels=lab)
    else:
        tmp[col_range],bins = pd.cut(tmp[col], cut_no,retbins=True,labels=lab)
    return pd.merge(tmp_df, tmp, how='left'), bins

def print_users_val_cnts(df,col):    
    col_range = col+'_級距' 
    if col_range not in df.columns:
        col_range = col
        
    print(df[['身分證字號',col_range]].drop_duplicates()[col_range].value_counts())


def cal_recall_from_test_df(df,df_test,by_groups,topn=10):
    '''計算只買一檔基金用戶(df_test)之推薦準確度recall
    ----
    df : 投資老手
    df_test : 投資新手(一檔基金)
    topn : 猜幾檔
    by_group: 按不同級距切出全行老手topn推薦基金
    '''
    pop_df_cols = getTopSellsFundsOrderByCols(df,by_groups).groupby(by_groups).head(topn)
    
#    cols = ['身分證字號','基金別'] + by_groups
#    df_test = df_test[cols].drop_duplicates()
    tmp_tst =pd.merge(df_test,pop_df_cols,how='left',on=by_groups)
    bool_guess = (tmp_tst.基金代碼_x ==tmp_tst.基金代碼_y)
    return bool_guess.sum() / df_test.身分證字號.nunique()


def build_cat_int_df(df,cat,cut_no=5,percentile=True):
    df,bins = cut_df(df, cat, cut_no=cut_no,percentile=percentile)
    print_users_val_cnts(df,cat)
    bins_df = pd.DataFrame({'名稱':cat+'_級距','級距':bins.astype('float32')})
    return df,bins_df

#%%
    
if __name__ == '__main__':
    '''資料讀取與db連結 '''
    yyyymmdd = pd.datetime.today().strftime('%Y%m%d')
#    con = pypyodbc.connect(
#        "DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=project2018"
#    )
    engine = create_engine("mssql+pyodbc://sa:01060728@dbm_public/project2018?driver=SQL Server")
    engine.connect()

    TOPN = 10
    ### 取出已整理貼標完成之基金購買紀錄 ##
    df = pd.read_sql('select * from v_基金推薦_投資戶標籤', con=engine) 
    
    '''投資戶:新手 + 老手'''        
    tmp_only1_uid = df.groupby(['身分證字號','基金代碼']).size()
    tmp_only1_uid = tmp_only1_uid.reset_index(level='基金代碼')
    num_of_funds_per_uid = tmp_only1_uid.groupby(tmp_only1_uid.index).size()
    only1_uid = np.array(num_of_funds_per_uid[num_of_funds_per_uid==1 ].index)
    
    df_new = df[df.身分證字號.isin(only1_uid)]
    df_old = df[~df.身分證字號.isin(only1_uid)]
    ''' 熱銷基金(排除新手) '''
    pop_df = getTopSellsFundsOrder(df_old)    
    

    # 分析以 ['aum級距','年齡級距','個人投資風險分數級距',...]作為分群依據的群間差異性
    #%%
    '''建立級距'''
    
    cats = ['當年度AP合計','當年度放款總AP','當年度台外幣總存款AP','當年度放款手續費',
            '當年度保險佣金金額(要保人)','最近六個月消費額','最近六個月刷卡數']
    cut_nos = [5, [0,0.7,0.9,0.99,1], [0,0.7,0.9,0.99,1],[0,0.995,0.999,0.9999,1],
               [0,0.7,0.9,0.99,1], [0,0.95,0.99,0.999,1],[0,0.7,0.9,0.999,1] ]
    
    bins_df = pd.DataFrame()
    for cat, cut_no in zip(cats,cut_nos):
        df_old, bin_df = build_cat_int_df(df_old,cat,cut_no)
        bins_df = bins_df.append(bin_df,ignore_index=True)
    
    #%%
    '''計算mae(比較不同變數級距下--群間差異性)'''
    gbs = [[e] for e in list(df_old.columns) if '級距' in e]
    gbs.append(['投資屬性'])
    gbs.append(['經管業務區'])
    mae_gbs = {}
    for gb in gbs:
        _, mae = cal_mae_df(df_old, gb, topn=TOPN, weight=True)
        mae_gbs[gb[0]] = mae
        if '_' not in gb[0]: ### 若不在bins_df裡面,寫入
            print(gb[0])
            bins_df = bins_df.append(
                    pd.DataFrame({'名稱':gb[0],'級距':df[gb[0]].unique()}),
                    ignore_index=True
                    )
    assert len(gbs) == bins_df.名稱.nunique(),'gbs, bins_df長度不一致'
    bins_df['yyyymmdd'] = yyyymmdd
    ####### 存入db 分切級距 ###########
    bins_df.to_sql('基金推薦_分切級距',con=engine,index=False,if_exists='append',
                   dtype = {'名稱':sqlalchemy.types.VARCHAR(length=255),
                            '級距':sqlalchemy.types.VARCHAR(length=100),
                            'yyyymmdd':sqlalchemy.types.VARCHAR(8)}
                   )
    
    '''存mae 入db '''

    mae_gb_df = pd.DataFrame({'mae': mae_gbs})
    mae_gb_df = mae_gb_df.reset_index().rename(columns={'index':'名稱'})
    mae_gb_df['yyyymmdd'] = yyyymmdd
    mae_gb_df['mae'] = mae_gb_df.mae.astype('float32')
    
    assert len(gbs) == mae_gb_df.shape[0], 'gbs, mae_gb_df 長度不一致'
    ## 存入 級距mae
    mae_gb_df.to_sql('基金推薦_級距mae', con=engine, index=False, if_exists='append',
                     dtype = {'名稱':sqlalchemy.types.VARCHAR(99),
                              'mae':sqlalchemy.types.DECIMAL(5,2),
                              'yyyymmdd':sqlalchemy.types.VARCHAR(8)
                             })	

    #%% recall
    '''RECALL :
        1. 計算benchmark --> 利用投資老手的熱銷基金,最為推薦給投資新手用戶 (~20%)
        2. 計算不同級距組合下,群組熱銷基金準確度
    '''
    
    new_id_buy = df_new.groupby(['身分證字號','基金代碼'])\
                .size()\
                .reset_index()\
                .rename(columns={0:'cnt'})
    benchmark = new_id_buy.基金代碼.isin(pop_df.基金代碼.head(10)).sum() / new_id_buy.身分證字號.nunique()
    print('pop recall:\n===========\n\trecall: {:.2f}%'.format(benchmark*100))
    
    add_cat_names = np.array([name for name in bins_df.名稱.unique() if '_' in name ])
    ap_y_bins = bins_df[bins_df.名稱 =='當年度AP合計_級距'].級距.values
    
    df_new['當年度AP合計_級距'] = pd.cut(df_new.當年度AP合計,
                                          bins=ap_y_bins, labels=range(len(ap_y_bins)-1))
    top10_fundid = pop_df.基金代碼.head(10)
    
    
    
    exclude =False ## 是否移除熱銷基金資料
    if exclude:
        df_exc = df_old.copy()
        df_exc = df_old[~df_old.基金代碼.isin(top10_fundid)]
        pop_exc = getTopSellsFundsOrder(df_exc).head(10)        
        benchmark_exc = new_id_buy.基金代碼.isin(pop_exc.基金代碼.head(10)).sum() / new_id_buy.身分證字號.nunique()
        print('pop recall:\n===========\n\trecall(exc pop): {:.2f}%'.format(benchmark_exc*100))

    else :
        df_exc = df_old
    

#
    ### df_new 會有重複購買相同基金之用戶 -->處理成df_test ####        
    cols = [['AUM級距'],['存款AUM占比級距'], ['年齡級距'],['投資屬性'],['經管業務區'],['當年度AP合計_級距']]
    recall_dict = {}
    
    df_test = df_new[[col[0] for col in cols] + ['身分證字號','基金代碼']].drop_duplicates()
    df_test = df_test[~df_test.基金代碼.isna()] ##排基金代碼= None
    
    for x,y,z,X in product(cols,cols,cols,cols):
        col_gb = list(set(x+y+z+X))
        print('==========================')
        gb_recall = cal_recall_from_test_df(df_exc,df_test,col_gb)
        print('group by:{}\nrecall:\t{:.2f} %'.format(col_gb,gb_recall*100))
        
        recall_dict['-'.join(col_gb)] = gb_recall
        
    recall_df = pd.DataFrame(
            {'recall':list(recall_dict.values())},
            index=recall_dict.keys()).drop_duplicates()
    recall_df = recall_df.reset_index().rename(columns={'index':'名稱'})
    
    recall_df.loc[-1] = ['top10',benchmark]
    recall_df.reset_index(drop=True,inplace=True)
    recall_df['yyyymmdd'] = yyyymmdd
    ### 存入db #####
    recall_df.to_sql('基金推薦_投資新手召回度',con=engine,
                     index=False, if_exists='append',
                     dtype={'名稱':sqlalchemy.types.VARCHAR(100),
                            'recall': sqlalchemy.types.DECIMAL(4,3),
                            'yyyymmdd': sqlalchemy.types.VARCHAR(8)
                             })
    
    #%% 利用群組熱銷當作推薦基金 
    '''用 [投資屬性,當年度ap合計_級距,年齡級距] 作為群組熱銷變數'''
    
    col_gb = ['投資屬性','當年度AP合計_級距','年齡級距']
    pop_by_colgb_df = getTopSellsFundsOrderByCols(df_old,col_gb).groupby(col_gb).head(10)
    pop_by_colgb_df['yyyymmdd']=yyyymmdd
    # 存入db 
    pop_by_colgb_df.to_sql('基金推薦_推薦清單_分組熱銷',con=engine, if_exists='append',index=False,
                           dtype={'投資屬性':sqlalchemy.types.SMALLINT,
                                  '基金代碼':sqlalchemy.types.VARCHAR(6),
                                  'yyyymmdd':sqlalchemy.types.VARCHAR(8),
                                  '當年度AP合計_級距':sqlalchemy.types.SMALLINT,
                                  '年齡級距': sqlalchemy.types.VARCHAR(99),
                                  '憑證數' : sqlalchemy.types.INT,
                                  '群組排名' : sqlalchemy.types.SMALLINT
                                   })
    #%%
    
    