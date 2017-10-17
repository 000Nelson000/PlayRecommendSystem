# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:54:01 2017

@author: 116952
"""

#%% 
import pandas as pd 
import pypyodbc 

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
# user- item iteaction from purchasing history
# =============================================================================



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
        rid_to_idx[idx] = rid
        idx_to_rid[rid] = idx
        
    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(cids):
        cid_to_idx[idx] = cid
        idx_to_cid[cid] = idx
        
    ### 
    
    
    
    
    


