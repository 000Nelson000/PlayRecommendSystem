# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:41:13 2017

@author: 116952
"""
# %% 
def get_itemids_ratings_np(model,predall):
    """retrive model rating , and itemids
    """
    num_users,num_items = predall.shape    
    rating = model.rating
    predall_itemid = np.zeros(num_users*num_items,dtype='object')
    for pos,e in enumerate(predall.flatten()):
        predall_itemid[pos] = idx_to_itemid[e]
    
    predall_itemid.shape = predall.shape
    predall_rating = np.sort(rating.A,axis=1,kind='heapsort')[:,:-model.topN-1:-1]
    return predall_itemid,predall_rating



def arrange_predict_to_dataframe(predall_itemids,predall_rating,
                                 model_kind):
    """arrange predicting rating to dataframe 
    """
    df_list = []
    for idx,(fund_ids,fund_scores) in tqdm(enumerate(\
                                        zip(predall_itemids,predall_rating)),
                                        total= predall_itemids.shape[0],
                                        unit = 'users'):
        userid = idx_to_userid[idx]
        df_list.append(pd.DataFrame({'userid' : userid,
                                     'fundid' : fund_ids,
                                     'score' : fund_scores,
                                     'model' : model_kind
                                     }))
    return pd.concat(df_list)
# %%
if __name__ == "__main__":
    from KNNmodel import *
    import numpy as np    
    from rec_helper import *
    from tqdm import tqdm    
    import pandas as pd    
    import scipy.sparse as sp          
    import sqlalchemy
    import pickle
    import time
    import pypyodbc    
    
    df_inter = pd.read_csv('./funds/purchase.csv',encoding='cp950')
    df_item = pd.read_csv('./funds/item_features.csv',encoding='cp950')
    df_user = pd.read_csv('./funds/user_features.csv',encoding='cp950')
    df_ufs = pd.read_csv('./funds/user_features_1.csv',encoding='utf8')
#    #%%
#    ### there are some fundids in df_inter not exists in df_item
    fundids_df_items = df_item['基金代碼'].as_matrix() # 1d array
    fundids_df_inter = df_inter['基金代碼'].unique() # 1d array
    fundids = np.intersect1d(fundids_df_inter,fundids_df_items) # 1d array
#    ##
    userids_crm1 = df_user['身分證字號'].unique()
    userids_crm2 = df_ufs['uid'].unique()
    userids = np.intersect1d(userids_crm1,userids_crm2)
    ## arrange purchasing data which fundid exist in fundids
    # (exclude data which is not exist in fundids)
    df_inter = df_inter.loc[df_inter['基金代碼'].isin(fundids)]
    df_inter = df_inter.loc[df_inter['身分證字號'].isin(userids)]
    ## user who bought at least two items
    df_gt2 = threshold_interaction(df_inter,'身分證字號','基金代碼',row_min=1,col_min=0) #
    ###
#    purchased_ui1, userid_to_idx1, \
#    idx_to_userid1, itemid_to_idx1,idx_to_itemid1= df_to_spmatrix(df_inter,'身分證字號','基金代碼')

    #train,test, user_idxs = train_test_split(purchased_ui,split_count=1,fraction=0.2)


#    purchased_ui, userid_to_idx, \
#    idx_to_userid, itemid_to_idx,idx_to_itemid = df_to_spmatrix(df_gt2,'身分證字號','基金代碼')
#    train,test, user_idxs = train_test_split(purchased_ui,split_count=1,fraction=0.2)
    # =============================================================================
    #      save to pickle -- 
    # =============================================================================
#    with open('./funds/sp_funds_datasets.pickle','wb') as f:
#        data = {
#                    'train':train,
#                    'test':test,
#                    'user_idxs':user_idxs, # test uids
#                    'idx_to_userid':idx_to_userid,
#                    'userid_to_idx':userid_to_idx,                    
#                    'itemid_to_idx':itemid_to_idx,
#                    'idx_to_itemid':idx_to_itemid
#                }
#        pickle.dump(data,f)
#    # =============================================================================
#    test_coo = test.tocoo()
#    uidx = test_coo.row
#    itemidx = test_coo.col
#    test_uids = [idx_to_userid[uid] for uid in uidx]
#    test_itemids = [idx_to_itemid[idx] for idx in itemidx] 
#    test_data = pd.DataFrame({'uid':test_uids,'itemid':test_itemids}) # 切出測試資料集
#    test_data.to_sql('ihong_基金推薦demo_測試資料',
#                     con = engine,index=False,dtype = {
#                         'uid':sqlalchemy.types.VARCHAR(length=12),
#                         'itemid':sqlalchemy.types.VARCHAR(length=12)
#                         }
#                     )    
    # =============================================================================
    #     load pickle -- train, test ,user_idx(test)
    # =============================================================================
    with open('./funds/sp_funds_datasets.pickle','rb') as f:
        data = pickle.load(f)
    test = data['test']
    train = data['train']
    user_idxs = data['user_idxs']
    idx_to_userid = data['idx_to_userid']
    userid_to_idx = data['userid_to_idx']
    idx_to_itemid = data['idx_to_itemid']
    itemid_to_idx = data['itemid_to_idx']
    
    ## users features
    df_item['iidx'] = df_item['基金代碼']\
        .apply(lambda row,mapper:mapper.get(row,np.nan), args=[itemid_to_idx])
    df_ufs['uidx'] = df_ufs['uid']\
        .apply(lambda row,mapper: mapper.get(row,np.nan), args=[userid_to_idx])
    # align with train data (user-row-based)
    df_ufs_align = df_ufs.sort_values(by='uidx').loc[:,:'d.AUM.300萬元以上']
    train_ufs = sp.csr_matrix(df_ufs_align)
    
    #%%
    # =============================================================================
    #  model
    # =============================================================================
    t1 = time.time()
    ## ibcf
    model_i = KNNmodel(train,kind='ibcf')
    model_i.jaccard_sim()
    model_i.fit(topK=100,remove=True)
    t2 = time.time()
    print('\n***'*10)
    print('time cost for ibcf:{:.1f}s'.format(t2-t1))
    ## ubcf
    t1 = time.time()
    model_u = KNNmodel(train,kind='ubcf')
    model_u.jaccard_sim()
    model_u.fit(topK=100,remove=True)
    t2 = time.time()
    print('\n***'*10)
    print('time cost for ubcf:{:.1f}s'.format(t2-t1))
    ## popular
    t1 = time.time()
    model_p = KNNmodel(train,kind='popular')
    model_p.fit(topK=100,remove=True)
    t2 = time.time()
    print('\n***'*10)
    print('time cost for popular:{:.1f}s'.format(t2-t1))
    ## ubcf-fs
    t1 = time.time()
    model_fs = KNNmodel(train,kind='ubcf_fs')
    model_fs.fit(topK=100,user_features = train_ufs,remove=True)
    t2 = time.time()
    print('\n***'*10)
    print('time cost for ubcf_fs:{:.1f}s'.format(t2-t1))
    #%%
    # =============================================================================
    # evaluate recall
    # =============================================================================
    uids = np.arange(0,train.shape[0])

    predall_u = model_u.predict(uids,topN=10)
    model_u.evaluate(predall_u,test,method='recall') # 24.09 (22.82 %)

    predall_i = model_i.predict(uids,topN=10)
    model_i.evaluate(predall_i,test,method='recall') # 11.72 (11.44 %)

    predall_p = model_p.predict(uids,topN=10)
    model_p.evaluate(predall_p,test,method='recall') # 20.14 (19.09 %)
    
    predall_ufs = model_fs.predict(uids,topN=10)
    model_fs.evaluate(predall_ufs,test,method='recall') # 31.95 (25.49 %)
    # =============================================================================
    # precision
    # =============================================================================
    model_u.evaluate(predall_u,test,method='precision') # 2.41 (2.28 %)
    model_i.evaluate(predall_i,test,method='precision') # 1.17 (1.14 %)
    model_p.evaluate(predall_p,test,method='precision') # 2.01 (1.91 %)
    model_fs.evaluate(predall_ufs,test,method='precision') # 3.20 (2.55 %)

    
    #%%
    predall_itemid_fs,predall_rating_fs = get_itemids_ratings_np(model_fs,predall_ufs)
    predall_itemid_u,predall_rating_u = get_itemids_ratings_np(model_u,predall_u)
    predall_itemid_p,predall_rating_p = get_itemids_ratings_np(model_p,predall_p)

    # =============================================================================
    # ### build df to sql ####
    # =============================================================================
    
    
    df = arrange_predict_to_dataframe(predall_itemid_fs,predall_rating_fs,'ubcf_fs') # ubcf-fs
    df2 = arrange_predict_to_dataframe(predall_itemid_u,predall_rating_u,'ubcf') # ubcf
    df3 = arrange_predict_to_dataframe(predall_itemid_p,predall_rating_p,'popular')# popular
    #df = arrange_predict_to_dataframe(predall_u_name,predall_u_rating,'ubcf')    
    #df2 = arrange_predict_to_dataframe(predall_itemid,predall_rating,model_fs.kind)
    #df_total = pd.concat([df,df2])
    
    df_total = pd.concat([df,df2,df3])
    df_total['rank'] = df_total.index +1
    #df_total
    
#    import pypyodbc
#    conn = pypyodbc.connect("DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=test")
    engine = sqlalchemy.create_engine("mssql+pyodbc://sa:01060728@dbm_public/test?driver=SQL Server")
    engine.connect()
    
    
    df_total['score'] = df_total['score'].astype('float16')
    
    df_total.to_sql('ihong_基金推薦demo_推薦清單', con=engine,index=False,
                    if_exists='replace',
                    dtype = {'fundid':sqlalchemy.types.VARCHAR(length=255),
                             'model':sqlalchemy.types.VARCHAR(length=20),
                             'score':sqlalchemy.types.DECIMAL(5,3),
                             'rank' : sqlalchemy.types.INT(),
                             'userid': sqlalchemy.types.VARCHAR(length=20)
                             })
    df_total.to_csv('./funds/recommended_list.csv',index=False) # save to csv local file
    
    df_gt2.to_sql('ihong_基金推薦demo_申購紀錄',con=engine,index=False,
                  if_exists='replace',
                  dtype = {'申購登錄年':sqlalchemy.types.SMALLINT,
                           '身分證字號':sqlalchemy.types.VARCHAR(length=20),
                           '基金中文名稱':sqlalchemy.types.VARCHAR(length=100),
                           '憑證':sqlalchemy.types.VARCHAR(length=20),
                           'db身分':sqlalchemy.types.VARCHAR(length=20),
                           '投資型態':sqlalchemy.types.VARCHAR(length=20),
                           '自然人身分':sqlalchemy.types.VARCHAR(length=20),
                           'idn': sqlalchemy.types.INT,
                           '基金代碼':sqlalchemy.types.VARCHAR(length=20),
                           '購買次數':sqlalchemy.types.SMALLINT,
                          })
    df_ufs.to_sql('ihong_基金推薦demo_用戶特徵',con=engine,index=False,
                  if_exists='replace',
                  dtype = {'國內股票型':sqlalchemy.types.SMALLINT,
                           '國外債券型':sqlalchemy.types.SMALLINT,
                           '國外股票型':sqlalchemy.types.SMALLINT,
                           'a.AUM.0元':sqlalchemy.types.SMALLINT,
                           'b.AUM.0.100萬元':sqlalchemy.types.SMALLINT,
                           'c.AUM.100.300萬':sqlalchemy.types.SMALLINT,
                           'd.AUM.300萬元以上':sqlalchemy.types.SMALLINT,
                           'uid':sqlalchemy.types.VARCHAR(length=20),
                           'uidx':sqlalchemy.types.SMALLINT
                          })
    
    df_item_used = df_item[df_item['基金代碼'].isin(fundids)]
    df_item_used['yyyymmdd'] = df_item_used['yyyymmdd'].astype('object')
    df_item_used['iidx'] = df_item_used['iidx'].astype('int')
    df_item_used.to_sql('ihong_基金推薦demo_基金特徵',con=engine,index=False,
                        if_exists='replace',
                        dtype={'更新時間':sqlalchemy.types.DATETIME,
                               'yyyymmdd':sqlalchemy.types.VARCHAR(length=8),
                               '基金代碼':sqlalchemy.types.VARCHAR(length=10),
                               '國內外基金註記':sqlalchemy.types.SMALLINT,
                               '基金規模(台幣/億)':sqlalchemy.types.DECIMAL(20,3),
                               '基金目前規模區間':sqlalchemy.types.VARCHAR(length=30),
                               '基金成立時間':sqlalchemy.types.DATETIME,
                               '基金成立幾年':sqlalchemy.types.SMALLINT,
                               '基金公司代碼':sqlalchemy.types.VARCHAR(length=10),
                               '計價幣別':sqlalchemy.types.VARCHAR(length=10),
                               '基金經理人':sqlalchemy.types.VARCHAR(length=200),
                               '區域別':sqlalchemy.types.VARCHAR(length=10),
                               '基金投資產業分類1':sqlalchemy.types.VARCHAR(length=50),
                               '基金投資產業分類2':sqlalchemy.types.VARCHAR(length=50),
                               '基金投資產業分類3':sqlalchemy.types.VARCHAR(length=50),
                               'aum基金型態別':sqlalchemy.types.VARCHAR(length=10),
                               '商品投資屬性':sqlalchemy.types.VARCHAR(length=10),
                               '高收益債註記':sqlalchemy.types.SMALLINT,
                               '保本型基金註記':sqlalchemy.types.VARCHAR(length=10),
                               '淨值':sqlalchemy.types.DECIMAL(20,2),
                               'sharpe':sqlalchemy.types.DECIMAL(5,2),
                               'beta':sqlalchemy.types.DECIMAL(5,2),
                               '一個月累積報酬率(%)':sqlalchemy.types.DECIMAL(20,2),
                               '三個月累積報酬率(%)':sqlalchemy.types.DECIMAL(20,2),
                               '六個月累積報酬率(%)':sqlalchemy.types.DECIMAL(20,2),
                               '一年累積報酬率(%)': sqlalchemy.types.DECIMAL(20,2),
                               '三年累積報酬率(%)':sqlalchemy.types.DECIMAL(20,2),
                               '五年累積報酬率(%)':sqlalchemy.types.DECIMAL(20,2),
                               '自今年以來報酬率(%)':sqlalchemy.types.DECIMAL(20,2),
                               '自成立日起報酬率(%)':sqlalchemy.types.DECIMAL(20,2),
                               '基金評等':sqlalchemy.types.DECIMAL(2,1),
                               '熱賣基金註記':sqlalchemy.types.SMALLINT,
                               '投資型態別':sqlalchemy.types.VARCHAR(length=20),
                               'cluster':sqlalchemy.types.SMALLINT,
                               'iidx':sqlalchemy.types.SMALLINT                         
                               })
    #%%
    # =============================================================================
    # item features (tag recommendated items some  "reason")
    # =============================================================================
    df_item_features = pd.read_sql("select * from v_ihong_基金推薦demo_基金特徵",conn)
    df_item_features.to_csv('./funds/df_item_features.csv',index=False)
    df_item_features = pd.read_csv('./funds/df_item_features.csv',sep=',',encoding='cp950')
    
    all_itemids = idx_to_itemid.values()
    bool_itemids = df_item_features['基金代碼'].isin(all_itemids)
    df_item_features = df_item_features[bool_itemids]
    
    #### establish items features look up table ######
    funds_f = {}
    for index, row in df_item_features.iterrows():
        temp_f = []        
        i_features = row.values
        iid = i_features[0]
        for idx, feat in enumerate(i_features):
            if feat!=None and str(feat)!='nan' and idx!= 0:                
                temp_f.append(feat)
        funds_f[iid] = temp_f
    
    all_uidxs = np.arange(train.shape[0])
    all_uids = [idx_to_userid[u] for u in all_uidxs]
    
    ##### find purchased items-features for a give userid ######
    ## df_gt2: this is all transaction data
    def get_features_given_uid(uid,df):
        """get features for a given uid and transaction data df        
        """
        purchased_fundids = df[df['身分證字號']==uid]['基金代碼'].unique().tolist()
        user_features = set()
        for fundid in purchased_fundids:
            user_features.update(funds_f[fundid])
            
        return user_features
    
#    user1_have_features = get_features_given_uid(all_uids[1],df_gt2)
    
    def get_recommended_item_for_user(itemid,have_features):
        """get common features between itemid and have features
        """
        fund_features = set(funds_f[itemid])
        return fund_features.intersection(have_features)

    get_recommended_item_for_user('F80',user1_have_features)
    
    ##### users have features ####
    users_have_features = {}
    for uid in tqdm(all_uids): ## 38 iter/sec --- pretty slow !!!
        users_have_features[uid] = get_features_given_uid(uid,df_gt2)
    ##### 
#    get_recommended_item_for_user('AE7',users_have_features['A1220335170'])
    df_total = df2
    
    def get_items_features(row):
        fundid = row['fundid']
        userid = row['userid']
        common_features_list = list(get_recommended_item_for_user(fundid,
                                      users_have_features[userid]))
        return ','.join(common_features_list)
#    get_items_features(fundids[1],userids[9])
    df_total['tag_features'] = df_total[['fundid','userid']].apply(get_items_features,axis=1)
    
            
