#! encoding: utf8
'''
整理
    user-features,
    item-features,
    u-i interaction
'''
import pandas as pd
import numpy as np
import pypyodbc
from rec_helper import *
from test_feat import *
from lightfm import LightFM
from lightfm.evaluation import recall_at_k, auc_score

if __name__ == '__main__':

    con = pypyodbc.connect(
        "DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=project2017")

    user_feat_df = pd.read_sql('select * from v_基金推薦_用戶特徵', con)
    item_feat_df = pd.read_sql('select * from 基金推薦_申購基金特徵', con)
    inter_df = pd.read_sql('select * from 基金推薦_近二年申購_憑證歸戶', con)

    ##### 針對所有用戶(有交易紀錄)建立sp-matrix #######
    # 38271*2075
    purchased_ui, userid_to_idx, \
        idx_to_userid, itemid_to_idx, idx_to_itemid = df_to_spmatrix(
            inter_df, '身分證字號', '基金代碼', binary=False)
    # item
    # 1. item id-> idx,
    # 2. 殘缺值補0
    iidx = item_feat_df['基金代碼'].map(lambda x: itemid_to_idx[x])
    not_in_iidx = np.where(
        ~np.isin(np.arange(len(itemid_to_idx)), iidx.values))[0]
    item_feat_df.set_index('基金代碼', inplace=True)
