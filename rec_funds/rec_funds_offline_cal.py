#! encoding: utf8

"""
利用協同過濾法 -- KNN相似度(jaccard)模型建立基金推薦清單(10筆/人/模型)
以近兩年的基金銷售歷史資料,做日批更新

=== 產出 ===
1. 評估推薦結果 (ibcf/ubcf/ubcf_fs/popular),       
2. 每個用戶推薦基金(topN)清單與推薦[假設性]原因
    
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

    # 申購紀錄

    df = pd.read_sql("select * from 基金推薦_近二年申購_憑證歸戶 ", con)

    # 建立u-i 矩陣 至少買過一檔基金
    df_gt2 = threshold_interaction(
        df, rowname='身分證字號', colname='基金代碼', row_min=1, col_min=0)

    purchased_ui, userid_to_idx, \
        idx_to_userid, itemid_to_idx, idx_to_itemid = df_to_spmatrix(
            df_gt2, '身分證字號', '基金代碼', binary=False)

    train, test, user_idxs = train_test_split(
        purchased_ui, split_count=1, fraction=0.2)

    ##
    uids_idx = df_gt2['身分證字號'].apply(map_ids, args=[userid_to_idx])
    iids_idx = df_gt2['基金代碼'].apply(map_ids, args=[itemid_to_idx])
    df_gt2['uidxs'] = uids_idx
    df_gt2['iidxs'] = iids_idx

    ### 測試集用戶屬性 (採購的基金屬性)
    train_coo = train.tocoo()
    train_uidxs, train_iidxs = train_coo.row, train_coo.col
    train_ui_list = list(zip(train_uidxs, train_iidxs))

    df_gt2['uidxs_iidxs'] = df_gt2['uidxs'].astype(
        'str') + '_' + df_gt2['iidxs'].astype('str')
    selected_ui_list = [str(e[0]) + '_' + str(e[1]) for e in train_ui_list]

    df_train = df_gt2.loc[df_gt2['uidxs_iidxs'].isin(selected_ui_list)]

    df_train.drop('uidxs_iidxs', axis=1)
    df_gt2.drop('uidxs_iidxs', axis=1)

    df_ui_features = df_train[['身分證字號', 'uidxs', 'iidxs', 'aum計算類別']]
    df_ui_features2 = df_ui_features.groupby(['uidxs', 'aum計算類別']).count()[
        ['身分證字號']].reset_index()
    df_ui_features2.columns = ['uidxs', '基金類別', '數量']
    df_ui_feats = df_ui_features2.pivot(
        index='uidxs', values='數量', columns='基金類別')

    # 取得AUM
    df_users_aum = pd.read_sql('select * from v_基金推薦_用戶特徵', con)

    df_users_aum_feats = df_users_aum.loc[df_users_aum['身分證字號'].isin(
        df_ui_features['身分證字號'])]

    temp = df_users_aum_feats['身分證字號'].apply(map_ids, args=[userid_to_idx])
    df_users_aum_feats = pd.concat(
        [df_users_aum_feats.iloc[:, 1:], temp.rename('uidxs')], axis=1)

    df_ui_feats['uidxs'] = df_ui_feats.index

    ## 合併用戶特徵(aum + 購買基金種類)
    users_feats = pd.merge(df_ui_feats.fillna(
        0), df_users_aum_feats, on='uidxs')
    users_feats = users_feats.sort_values(by='uidxs').drop('uidxs', axis=1)
    users_feats = users_feats.applymap(int)

    users_feats_sp = sp.csr_matrix(users_feats)

    users_feats['userid'] = users_feats.index.to_series().apply(
        map_ids, args=[idx_to_userid])

    return {
        'purchased_df' : df_gt2,
        'purchased_ui': purchased_ui,
        'userid_to_idx': userid_to_idx,
        'idx_to_userid': idx_to_userid,
        'itemid_to_idx': itemid_to_idx,
        'idx_to_itemid': idx_to_itemid,
        'eval_data': {
            'train': train,
            'test': test,
            'test_uidxs': user_idxs
        },
        'users_feats': {
            'df': users_feats,
            'sp': users_feats_sp
        }
    }


def model_eval(train, test, users_feats_sp):
    """評估以下模型在測試集準確度(`recall`)
    1. :ibcf : 以物品為基礎的相似度模型
    2. :ubcf : 以用戶為基礎的相似度模型
    3. :ubcf_fs: 利用特徵篩選的 ubcf
    4. :popular: 熱門物品

    params
    =====    
    `train` (sp.csr_matrix)

    `test`  (sp.csr_matrix)

    `users_feats_sp` (sp.csr_matrix) 
    """

    #### ibcf (jaccard) #####
    dt_ibcf, model_i = build_model(train, kind='ibcf', topK=100)
    #### ubcf (jaccard) #####
    dt_ubcf, model_u = build_model(train, kind='ubcf', topK=100)
    #### ubcf_fs (jaccard) #####
    dt_ufs, model_ufs = build_model(
        train, kind='ubcf_fs', topK=100, users_feats_sp=users_feats_sp)
    ### popular ####
    dt_pop, model_p = build_model(train, kind='popular', topK=100)

    #### 評估模型結果 ###
    uids = np.arange(0, train.shape[0])
    print('====== model evaluation (jaccard similarity) =====')
    predall_u = model_u.predict(uids, topN=10)  # np array (itemidx)
    model_u.evaluate(predall_u, test, method='recall')  # 29.27

    predall_i = model_i.predict(uids, topN=10)  # nparray (itemidx)
    model_i.evaluate(predall_i, test, method='recall')  # 12.71

    predall_p = model_p.predict(uids, topN=10)  # nparray (itemidx)
    model_p.evaluate(predall_p, test, method='recall')  # 20.08

    predall_ufs = model_ufs.predict(uids, topN=10)
    model_ufs.evaluate(predall_ufs, test, method='recall')

    recall_p = float("{:.4f}".format(model_p.recall))
    recall_i = float("{:.4f}".format(model_i.recall))
    recall_u = float("{:.4f}".format(model_u.recall))
    recall_ufs = float("{:.4f}".format(model_ufs.recall))

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    ## 寫入log ###
    with open('evaluation.log', 'a', encoding='utf8') as f:

        time_elapse = '''*************** Date: {0} *****************\n
        evaluation time ...

        --------------------------
        ibcf   : {1:^10},
        ubcf   : {2:^10},
        ubcf_fs: {3:^10},
        popular: {4:^10}
        --------------------------
        '''.format(now_str, dt_ibcf, dt_ubcf, dt_ufs, dt_pop)

        recall_eva = '''\n\t***** model evaluation *****\n
        numbers of users :{:^5}
        numbers of funds :{:^5}
        
        recall
        --------------------------
        popular:{:^10.1f}%
        ibcf:   {:^10.1f}%
        ubcf:   {:^10.1f}%
        ubcf_fs:{:^10.1f}%
        --------------------------
        '''.format(train.shape[0],train.shape[1],
                   recall_p * 100, recall_i * 100, recall_u * 100, recall_ufs * 100)

        f.write(time_elapse)
        f.write(recall_eva)

    model_lst = ['popular', 'ibcf', 'ubcf', 'ubcf_fs']
    recall_lst = [recall_p, recall_i, recall_u, recall_ufs]
    time_lst = [dt_pop, dt_ibcf, dt_ubcf, dt_ufs]

    return pd.DataFrame({
        'model': model_lst,
        'recall': recall_lst,
        'eval_time': time_lst,
        'num_users': train.shape[0],
        'num_funds': train.shape[1],
        'date': now.strftime('%Y%m%d')
    })


def build_model(sp_data, kind, topK=100, users_feats_sp=None):
    """計算並輸出knn model
    parmas
    =====
    `sp_data` (sparse matrix)

    `kind` (str) 

    `topK`

    `users_feats_sp` (sparse matrix) 
        - users sparse features
    return 
    ======
    `dt` (str) time elapse (h:m:s)
    `model` (KNNmodel)
    """
    t1 = time.time()
    model = KNNmodel(sp_data, kind=kind)
    if kind in ('ubcf', 'ibcf'):
        model.jaccard_sim()
    model.fit(topK=topK, user_features=users_feats_sp, remove=True)
    t2 = time.time()
    dt = int(t2 - t1)
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    dt = "{}:{:02}:{:02}".format(h, m, s)  # h:m:s
    print('****')
    print('time cost for building {} (jaccard) model: {}'.format(kind, dt))
    return dt, model


def recommender_lists(
        purchased_ui, idx_to_itemid, idx_to_userid, users_feats_sp):
    """基於交易資料建立推薦清單
    params
    =====
    `purchased_ui` (sp.csr_matrix)

    `idx_to_itemid` (dict)

    `idx_to_userid` (dict)

    `users_feats_sp` (sp.csr_matrix)
    """
    assert isinstance(purchased_ui, sp.spmatrix)

    ############## built recommender #####################
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    #### ibcf (jaccard)
    dt_ibcf, model_i = build_model(purchased_ui, kind='ibcf', topK=100)
    #### ubcf (jaccard)
    dt_ubcf, model_u = build_model(purchased_ui, kind='ubcf', topK=100)
    #### ubcf_fs (jaccard)
    dt_ufs, model_ufs = build_model(
        purchased_ui, kind='ubcf_fs', topK=100, users_feats_sp=users_feats_sp)
    # popular
    dt_pop, model_p = build_model(purchased_ui, kind='popular', topK=100)

    ############# recommender list ########################
    t0 = time.time()

    uids = np.arange(purchased_ui.shape[0])
    predall_i = model_i.predict(uids, topN=10)  # np array (itemidx)
    predall_u = model_u.predict(uids, topN=10)
    predall_ufs = model_ufs.predict(uids, topN=10)
    predall_p = model_p.predict(uids, topN=10)

    ############# 推薦標的分數/清單(numpy array) ############
    predall_itemid_i, predall_rating_i = get_itemids_ratings_np(
        model_i, predall_i, idx_to_itemid)

    predall_itemid_u, predall_rating_u = get_itemids_ratings_np(
        model_u, predall_u, idx_to_itemid)

    predall_itemid_ufs, predall_rating_ufs = get_itemids_ratings_np(
        model_ufs, predall_ufs, idx_to_itemid)

    predall_itemid_p, predall_rating_p = get_itemids_ratings_np(
        model_p, predall_p, idx_to_itemid)

    ############# 整理成 dataframe ############

    rec_ibcf_df = arrange_predict_to_dataframe(
        predall_itemid_i, predall_rating_i, 'ibcf', idx_to_userid)

    rec_ubcf_df = arrange_predict_to_dataframe(
        predall_itemid_u, predall_rating_u, 'ubcf', idx_to_userid)

    rec_ufs_df = arrange_predict_to_dataframe(
        predall_itemid_ufs, predall_rating_ufs, 'ubcf_fs', idx_to_userid)

    rec_pop_df = arrange_predict_to_dataframe(
        predall_itemid_p, predall_rating_p, 'popular', idx_to_userid)

    df_rec_total = pd.concat(
        [rec_ibcf_df, rec_ubcf_df, rec_ufs_df, rec_pop_df])
    df_rec_total['rank'] = df_rec_total.index + 1

    t1 = time.time()
    dt_arange_df = int(t1 - t0)  # sec
    m, s = divmod(dt_arange_df, 60)
    h, m = divmod(m, 60)
    dt_arange_df = "{}:{:02}:{:02}".format(h, m, s)  # h:m:s

    # write to log
    with open('build_recommender.log', 'a') as f:
        messages = '''*************** Date: {0} *****************\n
        recommender KNN model built for...
        ibcf   : {1:^5},
        ubcf   : {2:^5},
        ubcf_fs: {3:^5},
        popular: {4:^5}
        =============================
        arrange: {5:^5}
        '''.format(now_str, dt_ibcf, dt_ubcf, dt_ufs, dt_pop, dt_arange_df)
        f.write(messages)

    return df_rec_total


def map_ids(row, mapper):
    """用於`df.apply`的輔助函數 """
    return mapper[row]


def get_itemids_ratings_np(model, predall, idx_to_itemid):
    """retrive model rating, and itemids for given idxs
    parmas
    ======
    `model` : (KNNmodel)
        class of KNNmodel

    `predall` : (np.ndarray)
        predicted itemidx

    `idx_to_itemid` : (dict)    
        index of item to itemid lookup table

    return 
    ======
    `tuple`: (`predall_itemid`,`predall_rating`)

    `predall_itemid` : (np.ndarray)
        retrive itemid from idx_to_itemid

    `predall_rating`: (np.ndarray)
        predicted rating(scoring) corresponding to predall_itemid
    """
    num_users, num_items = predall.shape
    rating = model.rating
    predall_itemid = np.zeros(num_users * num_items, dtype='object')
    for pos, e in enumerate(predall.flatten()):
        predall_itemid[pos] = idx_to_itemid[e]

    predall_itemid.shape = predall.shape
    predall_rating = np.sort(rating.A, axis=1, kind='heapsort')[
        :, :-model.topN - 1:-1]
    return predall_itemid, predall_rating


def arrange_predict_to_dataframe(predall_itemids, predall_rating,
                                 model_kind, idx_to_userid):
    """arrange predicting itemids/rating to dataframe 
    params
    ======
    predall_itemids: (np.ndarray)
        predictions/recommendations w.r.t all ids preference
    predall_rating: (np.ndarray)
        predicted rating(scoring) corresponding to predall_itemid
    model_kind: (str)
        kind of model  (ubcf,ibcf,...)

    return
    ======
    dataframe
        - colname: 'userid','fundid','model','score'
    """
    df_list = []
    for idx, (fund_ids, fund_scores) in tqdm(enumerate(
            zip(predall_itemids, predall_rating)),
            total=predall_itemids.shape[0],
            unit='users'):
        userid = idx_to_userid[idx]
        df_list.append(pd.DataFrame({'userid': userid,
                                     'fundid': fund_ids,
                                     'score': fund_scores,
                                     'model': model_kind
                                     }))
    return pd.concat(df_list)


def save_df_to_msdb(con, df, tablename, data_append=True):
    """data.frame save into db
    """
    cursor = con.cursor()

    if not data_append:
        sql_delete = '''DELETE FROM {}'''.format(tablename)
        cursor.execute(sql_delete)
        cursor.commit()

    sql_insert = """INSERT INTO {0} ({1}) VALUES ({2})"""
    num_quest = '?' + ',?' * (df.shape[1] - 1)  # numbers of col

    for idx, row in df.iterrows():
        k, v = list(zip(*row.to_dict().items()))
        fields = ','.join(['[' + e + ']' for e in k])
        row = [None if pd.isnull(e) else e for e in v]
        cursor.execute(sql_insert.format(tablename, fields, num_quest), row)

    cursor.commit()


######## REASONING TAG HELPER ##########################################

def get_features_given_uid(uid, df, funds_f):
    """get features for a given uid and transaction data df
    
    params
    ======
    uid: (str)
        user id account
    df:(pd.dataframe)
        transaction data
    funds_f: (dict)
        features of funds lookup table
    return
    ======
    a set of users' features(tags)
    """
    purchased_fundids = df[df['身分證字號'] == uid]['基金代碼'].unique().tolist()
    user_features = set()
    for fundid in purchased_fundids:
        try:
            user_features.update(funds_f.get(fundid))
        except TypeError :
            print('No fundid:{} in mma ...'.format(fundid))
            continue

    return user_features


def get_recommended_item_for_user(itemid, have_features, funds_f):
    """get common features between itemid and have features
    
    params
    ======
    itemid: (str)
        itemid
    have_features: (set)
        user have features
    funds_f: (dict)
        features of funds lookup table
    
    return
    ======
    (set) 
    """
    fund_features = set(funds_f.get(itemid))
    return fund_features.intersection(have_features)

def built_items_features_lookup_table(df_item_features):
    '''establish items features look up table
    eg: 
        {
            '378':{
                    '區域_全球',
                    '月配',
                    '成熟股市',
                    ...
                }
        }
    params
    =====
    `df_item_features` (dataframe) 
        包含基金代碼的特徵矩陣
    return
    ======
    (dict) 基於**基金代碼**的特徵查找字典
    
    '''
    funds_f = {}
    for _, row in df_item_features.iterrows():
        row_dict = row.to_dict()
        iid = row_dict.pop('基金代碼')
        i_features = row_dict.values()
        i_feats_set = set.union(*[set(e.split('/'))
                                  for e in i_features if e != None])

        funds_f[iid] = i_feats_set
    return funds_f

def reason_tags(df_purchased, df_recommend, df_item_features):
    """給每個推薦的基金推薦**原因**

      --- 因為用戶買過的基金有xx屬性, oo也同樣有 ...

    eg:
     userA --> history: {a:('本國','高收益',...)}
                recommended : {c:('外國','高收益'),...}
                ==========
                reason: c = ('高收益')

    params
    ======
    `df_purchased` : (dataframe) 
        all transaction data (history)

    `df_recommend` : (dataframe) 
        all recommended items (based on uids)
        note--必須有`fundid`,`userid`欄位才能對應到

    `df_item_features`: (dataframe)
        items(funds) features table (base on fundids)
    """
    #### establish items features look up table ######
    funds_feat_lookup_dict = built_items_features_lookup_table(df_item_features)

    ##### users have features ####
    users_have_features = {}
    all_uids = df_purchased['身分證字號'].unique()
    for uid in tqdm(all_uids):  # 50 iter/sec --- pretty slow !!!
        users_have_features[uid] = get_features_given_uid(uid, df_purchased,funds_feat_lookup_dict)

    def get_items_features(row):
        fundid = row['fundid']
        userid = row['userid']
        common_features_list = list(get_recommended_item_for_user(fundid,
                                                                  users_have_features[userid],
                                                                  funds_feat_lookup_dict)
                                                                  )
        return ','.join(common_features_list)
    df_recommend['tag_features'] = df_recommend[['fundid','userid']].apply(get_items_features,axis=1)
    return df_recommend


if __name__ == '__main__':
    pass
    # con = pypyodbc.connect(
    #     "DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=project2017")

    # ###### evaluate model #######
    # data = load_data(con)
    # train = data['eval_data']['train']
    # test = data['eval_data']['test']
    # users_feats_sp = data['users_feats']['sp']
    # df_gt2 = data['purchased_df']
    # eval_result = model_eval(train, test, users_feats_sp)

    # ###### recommendation ######
    # purchased_ui = data['purchased_ui']
    # ###### features tag ########
    # df_item_features = pd.read_sql("select * from 基金推薦_申購基金特徵", con)
