#! encoding: utf8 

"""
以近兩年的基金銷售歷史資料,做日批更新
利用協同過濾法 -- KNN相似度(jaccard)模型建立基金推薦清單(10筆/人/模型)

=== 產出 ===
1. 評估結果 (ubcf/ubcf_fs/popular)
2. 每個用戶推薦基金與[假設性]原因
塞回ms資料庫
"""
import time
import datetime
import pypyodbc
import sys
import pandas as pd
import numpy as np 
import time
from KNNmodel import *
from rec_helper import *
from tqdm import tqdm
import scipy.sparse as sp


def load_data(con):
    """    
    讀取
    1. 近兩年交易資料
    2. 用戶特徵資料(aum + 購買基金屬性統計) 

    return 
    ======
    (JSON) 

    - `purchased_ui`  :(sp.csr_matrix) 原始交易ui資料 :至少購買一檔基金

    - `userid_to_idx` :(dict) 用戶id轉index :

    - `idx_to_userid` :(dict) 用戶index轉id  :

    - `itemid_to_idx` :(dict) 基金id轉index :

    - `idx_to_itemid` :(dict) 基金index轉基金id :

    - `eval_data` : 評估資料集 :    
      -  `train` :(sp.csr_matrix) 訓練ui資料  :

      -  `test`  :(sp.csr_matrix) 測試ui資料  : 從原始交易資料切出`20%`用戶並隨機挑選其中一檔基金作為測試

    - `users_feats` : 用戶特徵: 包含AUM級距與用戶申購之基金種類
      -  `df` : (dataframe):
      -  `sp` : (sp.csr_matrix):
     
    """

    ## 連接db
    
    df = pd.read_sql("select * from 基金推薦_近二年申購_憑證歸戶 ",con) ## 

    # 建立u-i 矩陣 至少買過一檔基金
    df_gt2 = threshold_interaction(df,rowname='身分證字號',colname='基金代碼',row_min=1,col_min=0) 

    purchased_ui, userid_to_idx, \
    idx_to_userid, itemid_to_idx,idx_to_itemid  = df_to_spmatrix(df_gt2,'身分證字號','基金代碼',binary=False) 
    
    train,test, user_idxs = train_test_split(purchased_ui,split_count=1,fraction=0.2)

    ##
    uids_idx = df_gt2['身分證字號'].apply(map_ids,args=[userid_to_idx])
    iids_idx = df_gt2['基金代碼'].apply(map_ids,args=[itemid_to_idx])
    df_gt2['uidxs'] = uids_idx
    df_gt2['iidxs'] = iids_idx

    ### 測試集用戶屬性 (採購的基金屬性)
    train_coo = train.tocoo()
    train_uidxs,train_iidxs = train_coo.row,train_coo.col
    train_ui_list = list(zip(train_uidxs,train_iidxs))

    df_gt2['uidxs_iidxs'] = df_gt2['uidxs'].astype('str') + '_'+ df_gt2['iidxs'].astype('str')
    selected_ui_list = [str(e[0]) + '_' + str(e[1]) for  e in train_ui_list]

    df_train = df_gt2.loc[df_gt2['uidxs_iidxs'].isin(selected_ui_list)]

    df_ui_features2 = df_ui_features.groupby(['uidxs','aum計算類別']).count()[['身分證字號']].reset_index()
    df_ui_features2.columns = ['uidxs','基金類別','數量']
    df_ui_feats = df_ui_features2.pivot(index='uidxs',values='數量',columns='基金類別')

    ## 取得AUM 
    df_users_aum = pd.read_sql('select * from v_基金推薦_用戶特徵',con)

    df_users_aum_feats = df_users_aum.loc[df_users_aum['身分證字號'].isin(df_ui_features['身分證字號'])]

    temp = df_users_aum_feats['身分證字號'].apply(map_ids,args=[userid_to_idx])
    df_users_aum_feats = pd.concat([df_users_aum_feats.iloc[:,1:],temp.rename('uidxs')],axis=1)

    ## 合併用戶特徵(aum + 購買基金種類)
    users_feats = pd.merge(df_ui_feats.fillna(0),df_users_aum_feats,on='uidxs')
    users_feats = users_feats.sort_values(by='uidxs').drop('uidxs',axis=1)
    users_feats_sp = sp.csr_matrix(users_feats)

    return {
                'purchased_ui'  : purchased_ui,
                'userid_to_idx' : userid_to_idx,
                'idx_to_userid' : idx_to_userid, 
                'itemid_to_idx' : itemid_to_idx,
                'idx_to_itemid' : idx_to_itemid,
                'eval_data' : {
                    'train' : train,
                    'test' : test,
                    'test_uidxs' : user_idxs
                },
                'users_feats' :{
                    'df' : users_feats,
                    'sp' : users_feats_sp
                }
            }



def model_eval(con,train,test,users_feats_sp):
    """評估以下模型在測試集準確度(`recall`)
    1. :ibcf : 以物品為基礎的相似度模型
    2. :ubcf : 以用戶為基礎的相似度模型
    3. :ubcf_fs: 利用特徵篩選的 ubcf
    4. :popular: 熱門物品

    params
    =====
    `con` (pypyodbc.connect)
    `train` (sp.csr_matrix)

    `test`  (sp.csr_matrix)

    `users_feats_sp` (sp.csr_matrix) 
    """

    #### ibcf (jaccard) #####
    t1 = time.time()
    model_i = KNNmodel(train,kind='ibcf')
    model_i.jaccard_sim()
    model_i.fit(topK=100,remove=True)
    t2 = time.time()
    dt_ibcf = t2-t1
    print('***')
    print('model built for ibcf:{:.1f}s'.format(dt_ibcf))
    
    #### ubcf (jaccard) #####    
    t1 = time.time()
    model_u = KNNmodel(train,kind='ubcf')
    model_u.jaccard_sim()
    model_u.fit(topK=100,remove=True)
    t2 = time.time()
    dt_ubcf = t2-t1
    print('\n***')
    print('model built for ubcf:{:.1f}s'.format(dt_ubcf))

    #### ubcf_fs (jaccard) #####    
    t1 = time.time()
    model_ufs = KNNmodel(train,kind='ubcf_fs')
    model_ufs.fit(topK=100,remove=True,user_features= users_feats_sp)
    t2 = time.time()
    dt_ufs = t2-t1
    print('\n***')
    print('model built for ubcf:{:.1f}s'.format(dt_ufs))
    
    ### popular ####
    t1 = time.time()
    model_p = KNNmodel(train,kind='popular')
    model_p.fit(topK=100,remove=True)
    t2 = time.time()
    dt_pop = t2-t1
    print('\n***')
    print('model built for popular:{:.1f}s'.format(dt_pop))
    
    #### 評估模型結果 ###
    uids = np.arange(0,train.shape[0])
    print('====== model evaluation (jaccard similarity) =====')
    predall_u = model_u.predict(uids,topN=10) # np array (itemidx)
    model_u.evaluate(predall_u,test,method='recall') # 29.27

    predall_i = model_i.predict(uids,topN=10) #nparray (itemidx)
    model_i.evaluate(predall_i,test,method='recall') # 12.71 

    predall_p = model_p.predict(uids,topN=10) #nparray (itemidx)
    model_p.evaluate(predall_p,test,method='recall') # 20.08
    
    predall_ufs = model_ufs.predict(uids,topN=10)
    model_ufs.evaluate(predall_ufs,test,method='recall')

    recall_p = model_p.recall
    recall_i = model_i.recall
    recall_u = model_u.recall
    recall_ufs = model_ufs.recall

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    with open('evaluation_.log','a') as f:
        
        time_elapse = '''***** Date: {0}*****
        model built for ibcf   :{1:.1f}s,
        model built ubcf   :{2:.1f}s,
        model built ubcf_fs:{3:.1f}s,
        model built popular:{4:.1f}s
        '''.format(now_str,dt_ibcf,dt_ubcf,dt_ufs,dt_pop)

        recall = '''\n-----precision-----\n\tpopular:{0:^10.1f}%\n\tibcf:{1:^10.1f}%\n\tubcf:{}'''

        f.write(time_elapse)
        f.write()
    



def recommender_lists():
    pass

def map_ids(row,mapper):
    return mapper[row]

if __name__ == '__main__':

    con = pypyodbc.connect("DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=project2017")

