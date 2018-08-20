#!encoding : utf8

'''此腳本利用級距切出不同族群之用戶並分析其群間差異性(分析用),
    ex: 性別(男女)族群與全體熱銷基金的差異程度
    
        男    女    全體
        1     3     1
        5     1     2
        4     2     3
        2     4     4
        3     5     5
---------------------------------------------------
* 
    - 老手客戶 
    - 資料來源 : v_ihong_基金推薦demo_基金用戶貼標 (dbo.test)
* 目的
    - 計算在不同面向/變數的熱銷基金，其群間差異性    
    - 計算在不同面向/變數的熱銷推薦基金，其推薦召回度(recall)
* 原理
    - 群間差異性定義為:  
        - 與全用戶熱銷基金(top10)的 mean absolute error(mae)
---------------------------------------------------
'''
#%%
import pandas as pd
import numpy as np
import pypyodbc
from itertools import product
from collections import defaultdict

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
        
    return df[['身分證字號',col_range]].drop_duplicates()[col_range].value_counts()


def cal_recall_from_test_df(df,df_test,by_groups,topn=10):
    '''計算只買一檔基金用戶(df_test)之推薦準確度recall
    ----
    df : 全行用戶
    df_test : 測試用戶
    topn : 猜幾檔
    by_group: 按不同級距切出全行老手topn推薦基金
    '''
    pop_df_cols = getTopSellsFundsOrderByCols(df,by_groups).groupby(by_groups).head(topn)
    
    cols = ['身分證字號','基金別'] + by_groups
#    df_test = df_test[cols].drop_duplicates()
    tmp_tst =pd.merge(df_test,pop_df_cols,how='left',on=by_groups)
    bool_guess = (tmp_tst.基金別 ==tmp_tst.基金代碼)
    return bool_guess.sum() / df_test.身分證字號.nunique()



#%%
if __name__ == '__main__':
    
    con = pypyodbc.connect(
        "DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=test"
    )
    TOPN = 10
    ### 取出已整理貼標完成之基金購買紀錄 ##
    df = pd.read_sql('select * from v_ihong_基金推薦demo_基金用戶貼標', con)
    ### 熱銷基金 #####
    pop_df = getTopSellsFundsOrder(df)    
    

    # 分析以 ['aum級距','年齡級距','個人投資風險分數級距',...]作為分群依據的群間差異性
    #%%
    ##### 建立級距:
    ###### ap #######
    df,ap_y_bins = cut_df(df, '當年度ap合計', cut_no=5,percentile=True)
    print_users_val_cnts(df,'當年度ap合計')
    
    df,ap_loan_bins = cut_df(df,'當年度放款總ap',[0,0.7,0.9,0.99,1],percentile=True) ## 0:22817, (0,4171]:872, (4171,103227]:2369, (103227,1063778] 264    
    print_users_val_cnts(df,'當年度放款總ap')

    df,ap_for_bins = cut_df(df, '當年度台外幣總存款ap',cut_no=[0,0.7,0.9,0.99,1],percentile=True) # 
    print_users_val_cnts(df,'當年度台外幣總存款ap')
    
    df,fee_for_bins = cut_df(df,'當年度放款手續費',cut_no=[0,0.995,0.999,0.9999,1],percentile=True)  ###
    print_users_val_cnts(df,'當年度放款手續費')
    
    df,fee_ins_bins = cut_df(df,'當年度保險佣金金額(要保人)',cut_no=[0,0.7,0.9,0.99,1]) ### 
    print_users_val_cnts(df,'當年度保險佣金金額(要保人)')
    
    ## 信用卡 #######
    df,amt_cc_bins = cut_df(df, '最近六個月消費額',[0,0.95,0.99,0.999,1])
    print_users_val_cnts(df,'最近六個月消費額')
    
    df,no_cc_bins = cut_df(df, '最近六個月刷卡數',[0,0.7,0.9,0.999,1],percentile=True) ##[0,14] -- 18647, (14,57] 5057, (57,573.11] 2591,(573,2247] 27
    print_users_val_cnts(df,'最近六個月刷卡數')
    ## 證券 ########
    df,amt_stock_bins = cut_df(df, '證券六月交易額',[0,0.94,0.95,0.96,0.99,0.999,1])
    print_users_val_cnts(df,'證券六月交易額')
    
    df,no_stock_bins = cut_df(df,'證券6月交易筆數',[0,0.94,0.95,0.96,0.99,0.999,1])
    print_users_val_cnts(df,'證券6月交易筆數')

    #%%
#    gbs = [
#        ['aum級距'],
#        ['年齡級距'],
#        ['個人投資風險分數級距'],
#        ['存款aum占比級距'],
#        
#    ]
    gbs = [[e] for e in list(df.columns) if '級距' in e]
    gbs.append(['投資屬性'])
    gbs.append(['經管業務區'])
    mae_gbs = {}
    for gb in gbs:
        _, mae = cal_mae_df(df, gb, topn=TOPN, weight=True)
        mae_gbs[gb[0]] = mae

    mae_gb_df = pd.DataFrame({'mae': mae_gbs})
    #%% 將top10視為離群(偽造資料)...刪除
    top10_fundid = pop_df.基金代碼.head(10)
    df_exc = df.copy()
    df_exc = df_exc[~df_exc.基金代碼.isin(top10_fundid)]
    
    mae_gbs_exc = {}
    for gb in gbs:
        _, mae = cal_mae_df(df_exc, gb, topn=TOPN, weight=True)
        mae_gbs_exc[gb[0]] = mae

    mae_gb_exc_df = pd.DataFrame({'mae': mae_gbs_exc})
    #%% 切出2個組合變數 
    '''mae @ feat1 x feat2 (2d)'''
    feat = [gb[0] for gb in gbs]
    mae_2d= np.zeros(shape=(len(feat),len(feat))) ## preallocate 
    mae_2d_exc= np.zeros(shape=(len(feat),len(feat))) ## preallocate 
    for idx,(x_feat,y_feat) in enumerate(product(feat,feat)):
        row = idx //len(feat) ; col = idx % len(feat)
        if x_feat == y_feat :
            gb = [x_feat]
        else:
            gb = [x_feat,y_feat]
        _, mae_group = cal_mae_df(df,gb,weight=True)
        _, mae_group_exc = cal_mae_df(df_exc,gb,weight=True)
        print('gb {:.2f}: group {},{}'.format(mae_group,x_feat,y_feat))

        mae_2d[row,col] = mae_group
        mae_2d_exc[row,col] = mae_group_exc
    mae2d_df = pd.DataFrame(mae_2d,columns=feat,index=feat)
    mae2d_df_exc = pd.DataFrame(mae_2d_exc,columns=feat,index=feat)

    cols = [x + '_' + y for x,y in product(feat,feat)]
    mae_2_df = pd.DataFrame({'mae':mae_2d.ravel()},index=cols).drop_duplicates()
    mae_2_df_exc = pd.DataFrame({'mae':mae_2d_exc.ravel()},index=cols).\
                    drop_duplicates().sort_values('mae',ascending=False).head(20)
    #%% 切n=3個組合變數
    ''' mae @ feat1 x feat2 x feat3'''
    
    feat = [gb[0] for gb in gbs]
    mae_3 = defaultdict(float)
    for idx,(x_feat,y_feat,z_feat) in enumerate(product(feat,feat,feat)):
        
        set_xyz = {x_feat,y_feat,z_feat}
        gb = list(set_xyz)        
            
        _, mae_group = cal_mae_df(df,gb,weight=True)
        print('gb {:.2f}: group {}'.format(mae_group,gb))
        mae_3[x_feat + '_' + y_feat + '_' + z_feat] = mae_group
    
    mae_3_df = pd.DataFrame({'mae':list(mae_3.values())}, 
                             index=list(mae_3.keys())).drop_duplicates().\
                             sort_values(by='mae',ascending=False)
    #%% 利用群組熱銷當作推薦基金 與test 資料回測比較recall
   
    
    df_test = pd.read_sql('''select * from v_ihong_申購一檔基金用戶名單''',con)
    
    # benchmark : pop
    benchmark = df_test.基金別.isin(pop_df.基金代碼.head(10)).sum() / df_test.身分證字號.nunique() ## 4047
    print('pop recall:\n===========\n\trecall: {:.2f}%'.format(benchmark*100))

#    gb_recall = cal_recall_from_test_df(df,df_test,cols_gb)
#    print('group by:{}\n===========\nrecall:\t{:.2f} %'.format(cols_gb,gb_recall*100))
    df_test['當年度ap合計_級距'] = pd.cut(df_test.當年度ap合計, 
                                   bins=ap_y_bins,labels=range(len(ap_y_bins)-1))
    
    top10_fundid = pop_df.基金代碼.head(10)
    
    exclude =False ## 是否移除熱銷基金資料
    if exclude:
        df_test = df_test[~df_test.基金別.isin(top10_fundid)]
        df_exc = df.copy()
        df_exc = df_exc[~df_exc.基金代碼.isin(top10_fundid)]
        pop_exc = getTopSellsFundsOrder(df_exc).head(10)
        benchmark = df_test.基金別.isin(pop_exc.基金代碼.head(10)).sum() / df_test.身分證字號.nunique() ## 4047
        print('pop recall:\n===========\n\trecall: {:.2f}%'.format(benchmark*100))

    else :
        df_exc = df
    


        
    cols = [['aum級距'],['存款aum占比級距'], ['年齡級距'],['投資屬性'],['經管業務區'],['當年度ap合計_級距']]
    recall_dict = {}
    for x,y,z,X in product(cols,cols,cols,cols):
        col_gb = list(set(x+y+z+X))
        print('==========================')
        gb_recall = cal_recall_from_test_df(df_exc,df_test,col_gb)
        print('group by:{}\nrecall:\t{:.2f} %'.format(col_gb,gb_recall*100))
        
        recall_dict['-'.join(col_gb)] = gb_recall
        
    recall_df = pd.DataFrame(
            {'recall':list(recall_dict.values())},
            index=recall_dict.keys()).drop_duplicates()
    