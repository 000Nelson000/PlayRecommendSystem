# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:54:01 2017

@author: 116952
"""

#%% 
import pandas as pd 
import scipy.sparse as sp
import pypyodbc 
import numpy as np 
from lightfm import LightFM
from lightfm.evaluation import auc_score,recall_at_k,precision_at_k

con = pypyodbc.connect("DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=xxxxxxx;DATABASE=project2017")

#%% 
sql_query = """\n
select * from v_基金推薦_申購明細 where [申購登錄年] >= 2015 """
df = pd.read_sql(sql_query,con)

sql_itemfeatures = """select * from project2017.dbo.v_基金推薦_基金屬性 """
df_item = pd.read_sql(sql_itemfeatures,con)

sql_userfeatures = """ select  * from Test.dbo.Shane_基金推薦_XGBoost"""
df_user = pd.read_sql(sql_userfeatures,con)
#%%

# =============================================================================
# save raw-data to csv 
# =============================================================================
df.to_csv('./funds/purchase.csv',index=False)
df_item.to_csv('./funds/item_features.csv',index=False)
df_user.to_csv('./funds/user_features.csv',index=False)

#%%
# =============================================================================
# Load csv data to memory
# =============================================================================

df_inter = pd.read_csv('./funds/purchase.csv',encoding='cp950')
df_item = pd.read_csv('./funds/item_features.csv',encoding='cp950')
df_user = pd.read_csv('./funds/user_features.csv',encoding='cp950')

#%% 

# =============================================================================
# user- item iteaction from purchasing history
# =============================================================================
df_gt2 = threshold_interaction(df_inter,'身分證字號','基金代碼')
purchased_ui, userid_to_idx, \
idx_to_userid, itemid_to_idx,idx_to_itemid = df_to_spmatrix(df_inter,'身分證字號','基金代碼')
train,test, user_idxs = train_test_split(purchased_ui,split_count=1,fraction=0.2)

#%% 
def threshold_interaction(df,rowname,colname,row_min=1):
    """limit interaction(u-i) dataframe greater than row_min numbers 
    
    Parameters
    ----------
    df: Dataframe
         purchasing dataframe         
         
    rowname: 
        name of user(Uid)
    colname: 
        name of item(Iid)
        
    """
    row_counts = df.groupby(rowname)[colname].count()
#    col_counts = df.groupby(colname)[rowname]
    uids = row_counts[row_counts > row_min].index.tolist() # lists of uids purchasing item greater than row_min
    
    df2 = df[df[rowname].isin(uids)]
    return df2

def df_to_spmatrix(df, rowname, colname):
    """convert dataframe to sparse (interaction) matrix
    
    Pamrameters
    -----------
    df: Dataframe
        
    Returns:
    -----------
    interaction : sparse csr matrix
        
    rid_to_idx : dict
        Map row ID to idx in the interaction matrix
    idx_to_rid : dict
    cid_to_idx : dict
    idx_to_cid :dict
    """
    rids = df[rowname].unique().tolist() # lists of rids 
    cids = df[colname].unique().tolist() # lists of cids
    
    ### map row/column id to idx ###
    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(rids):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid
        
    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(cids):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid
        
    ### 
    
    def map_ids(row, mapper):
        return mapper[row]
    
    I = df[rowname].apply(map_ids, args=[rid_to_idx]).as_matrix()
    J = df[colname].apply(map_ids, args=[cid_to_idx]).as_matrix()
    V = np.ones(I.shape[0])    
    interactions = sp.coo_matrix((V,(I,J)),dtype='int32')
    interactions = interactions.tocsr()
    
    return interactions,rid_to_idx,idx_to_rid,cid_to_idx,idx_to_cid


def train_test_split(sp_interaction, split_count, fraction=None):
    """ split interaction data into train and test sets 
    
    Parameters
    ----------
    sp_interaction: csr sparse matrix
    split_count: int
        Number of u-i interaction per user to move from original sets 
        to test set.
    fractions: float
        Fractions of users to split off into test sets.
        If None, all users are considered. 
    """
    train = sp_interaction.copy().tocoo()
    test = sp.lil_matrix(train.shape)  
    
    if fraction:
        try:
            user_idxs = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_idxs = range(sp_interaction.shape[0])
        
    train = train.tolil()
    for uidx in user_idxs:
        test_interactions = np.random.choice(sp_interaction.getrow(uidx).indices,
                                        size=split_count,
                                        replace=False)
        train[uidx, test_interactions] = 0.
        test[uidx, test_interactions] = sp_interaction[uidx, test_interactions]
        
    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_idxs

#%%
# =============================================================================
# train model, no- features used ( naive test ----)
# =============================================================================
NUM_THREADS =4
NUM_COMPONENTS = 30
model = LightFM(loss='warp',
                no_components=NUM_COMPONENTS,
                random_state=2017)
model.fit(train,epochs=20,
          num_threads=4,verbose=True)


# sanity check
# Don't forget the pass in the item features again!
train_auc = auc_score(model,
                      train,                      
                      num_threads=NUM_THREADS).mean()
train_recall = recall_at_k(model,
                           train,
                           num_threads=NUM_THREADS).mean()
print('sanity check\n\tno-features used trained auc: {0:.2f}\n'.format(train_auc))
print('\tno-features used trained recall: {0:.2f}\n'.format(train_recall))

#%%
# test check 
test_auc = auc_score(model,
                    test,
#                    train_interactions=train,
#                    item_features=item_features,
                    num_threads=NUM_THREADS).mean()
test_recall = recall_at_k(model,
                          test,
                          num_threads=NUM_THREADS).mean()
print('TEST===\n\tauc : {0:.2f}\n'.format(test_auc))
print('\t recall: {0:.2f}'.format(test_recall))

# =============================================================================
# grid search
# =============================================================================
