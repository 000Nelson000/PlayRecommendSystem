# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:54:01 2017

@author: 116952
"""

#%% 
import itertools
import pandas as pd 
import scipy.sparse as sp
import pypyodbc 
import numpy as np 
import lightfm
import copy 
import pickle
from lightfm import LightFM
from lightfm.evaluation import auc_score,recall_at_k,precision_at_k


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

def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-'*spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif (isinstance(r, float)
              or isinstance(r, np.float32)
              or isinstance(r, np.float64)):
            middle += '| {0:^{1}.5f} '.format(r, spacing-2)
        bottom += '+{}'.format('='*spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)


#def learning_curve_implicit(model, train, test, epochs, k=5, user_index=None):
#    if not user_index:
#        user_index = range(train.shape[0])
#    prev_epoch = 0
#    train_precision = []
#    train_mse = []
#    test_precision = []
#    test_mse = []
#    
#    headers = ['epochs', 'p@k train', 'p@k test',
#               'mse train', 'mse test']
#    print_log(headers, header=True)
#    
#    for epoch in epochs:
#        model.iterations = epoch - prev_epoch
#        if not hasattr(model, 'user_vectors'):
#            model.fit(train)
#        else:
#            model.fit_partial(train)
#        train_mse.append(calculate_mse(model, train, user_index))
#        train_precision.append(precision_at_k(model, train, k, user_index))
#        test_mse.append(calculate_mse(model, test, user_index))
#        test_precision.append(precision_at_k(model, test, k, user_index))
#        row = [epoch, train_precision[-1], test_precision[-1],
#               train_mse[-1], test_mse[-1]]
#        print_log(row)
#        prev_epoch = epoch
#    return model, train_precision, train_mse, test_precision, test_mse

def learning_curve(model, train, test, eval_train,
                        iterarray, user_features=None,
                        item_features=None, k=5,
                        **fit_params):
    """calculate learning curve for a lightfm reccommender
    
    parameters
    ----------
    
    model: LightFM model    
            
    train: csr matrix
        training set (indcluding all users)        
    eval_train: csr matrix
        evaluate set (remove all non-evaluate<not in test> users' data) 
    test: csr matrix 
        test set (provide ans in a eval_train users' sets)
    iterarray: list
        numbers of epochs to be evaluated
    k: int
        numbers of recommended items
    user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
        Each row contains that user’s weights over features.
    item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
        Each row contains that item’s weights over features.    
        
    return
    ------
    model: LightFM 
        reccommender model     
    train_patk: float
        precision at k in (eval_)training data 
    test_patk: float
        precision at k in testing data
    train_ratk: float
        recall at k in (eval_)training data
    test_ratk: float
        recall at k in testing data
    train_rrk: float
        reciprocal_rank at k in (eval_)training data
    test_rrk: float
        reciprocal_rank at k in test data
    """
    old_epoch = 0
    
    train_patk = [] # precision at k (train)
    test_patk = [] # precision at k (eval- train)
    train_ratk = [] # recall at k (eval train)
    test_ratk = [] # recall at k (test)
    train_rrk = [] # reciprocal rank (eval train)
    test_rrk = [] # reciprocal rank (test)
    
    
    headers = ['Epoch', 
               'train p@' + str(k), 
               'test p@' + str(k), 
               'train r@' + str(k),
               'test r@' + str(k),
               'train rr@' + str(k),
               'test rr@' + str(k)
               ]
    
    print_log(headers, header=True)
    
    for epoch in iterarray:
        more = epoch - old_epoch
        model.fit_partial(train, 
                          user_features=user_features,
                          item_features=item_features,
                          epochs=more, **fit_params)
        ## precision at k 
        this_test_pk = lightfm.evaluation.precision_at_k(model, test, train_interactions=None, k=k)
        this_train_pk = lightfm.evaluation.precision_at_k(model, eval_train, train_interactions=None, k=k)
        train_patk.append(np.mean(this_train_pk)) # store into list
        test_patk.append(np.mean(this_test_pk))
        
        ## recall at k 
        this_test_rk = lightfm.evaluation.recall_at_k(model,test,k=k)
        this_train_rk = lightfm.evaluation.recall_at_k(model,eval_train, k=k)
        train_ratk.append(np.mean(this_train_rk))
        test_ratk.append(np.mean(this_test_rk))
                
        ## reciprocal_rank at k
        this_test_rrk = lightfm.evaluation.reciprocal_rank(model,test)
        this_train_rrk = lightfm.evaluation.reciprocal_rank(model,eval_train)
        
        train_rrk.append(np.mean(this_train_rrk))
        test_rrk.append(np.mean(this_test_rrk))
        ## print log 
        row = [epoch, 
               train_patk[-1],
               test_patk[-1], 
               train_ratk[-1], 
               test_ratk[-1],
               train_rrk[-1],
               test_rrk[-1]
               ]
        
        print_log(row)
        
        old_epoch = epoch
        
    return model, train_patk, test_patk, train_ratk, test_ratk, train_rrk,test_rrk,

def popularity_guess(train,topn=10,popular_n=100):        
    """find the most popular item for each user
    
    return
    ------
    popular_array: np.array (dim: num_users*topn)
        top n most popular items user not bought before
    """
    train_lil = train.tolil()
    ## time comsuming
    scores_items = np.squeeze(np.asarray(train_lil.astype('int32').sum(axis=0)))
    popular_cand = scores_items.argsort()[-popular_n:][::-1]
    
    popular_array = np.zeros((train_lil.shape[0],topn),dtype=int)
    rows_itemIdx = train_lil.rows  
    for user,nz_idx in enumerate(rows_itemIdx):
#        bool_pop = [e not in nz_idx for e in popular_list]
        idx_set = set(popular_cand) - set(nz_idx)
        bool_cand = np.array([e in idx_set for e in popular_cand])
        popular_array[user,] = popular_cand[bool_cand][:topn]

    return popular_array


def eval_popularity(popularity,eval_train,test,method='recall'):
    """evaluate precision/recall for top n popular items
    """
    eval_train_lil = eval_train.tolil()
    all_train_items = eval_train_lil.rows
    
    test_lil = test.tolil()
    all_test_items = test_lil.rows
    hit = 0
    num_user = 0
    if method == 'recall':
        for user,rows in enumerate(all_train_items):
            if rows: # not None
                s1 = set(all_test_items[user]) # test result
                s2 = set(popularity[user]) # pop guess
                if s1.intersection(s2):
                    hit += 1
                num_user += 1
        recall = hit / num_user
        
    elif method == 'precision':
        pass
    
    return recall

def objective(params,method = 'precision'):
    """obective function we want to minimize w.r.t hyperparameters
    
    parameters
    ----------
    params: tuple of 4, float
        epochs,learning rate, no_compnents, alpha 
    method: 'precision'(default) or 'recall'
        metric we want to minimize
    returns
    -------
    objective value we want to minimze 
    """
    epochs,learning_rate,\
    no_components, alpha = params
    
    user_alpha = alpha
    item_alpha = alpha
    
    model = LightFM(loss = 'warp',
                    random_state = 2017,
                    learning_rate = learning_rate,
                    no_components = no_components,
                    user_alpha = user_alpha,
                    item_alpha = item_alpha)
    
    model.fit(train, epochs=epochs, num_threads=4,
              verbose=True)
    if method == 'precision':
        ## precision at k
        score_array = lightfm.evaluation.precision_at_k(model,
                                                   test,
                                                   train_interactions=None,
                                                   k=5,
                                                   num_threads=4)
    elif method == 'recall':
        ## recall at k 
        score_array = lightfm.evaluation.recall_at_k(model,
                                                test,
                                                train_interactions=None,
                                                k=5,
                                                num_threads=4)
    else:
        raise  
    ## we want to min
    out = -np.mean(score_array)
    
    # handle some weird numerical shit result
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out
        
#def grid_search_learning_curve(base_model, train, test, eval_train,param_grid,
#                               user_index=None, atk=5, epochs=range(2, 40, 2)):        
#    """grid search 
#    
#    "Inspired" (stolen) from sklearn gridsearch
#    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
#    
#    """
#    curves = []
#    keys, values = zip(*param_grid.items())
#    for v in itertools.product(*values):
#        params = dict(zip(keys, v))
#        this_model = copy.deepcopy(base_model)
#        print_line = []
#        for k, v in params.items():
#            setattr(this_model, k, v)
#            print_line.append((k, v))
#
#        print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
#        _, train_patk, train_mse, test_patk, test_mse = learning_curve(this_model, train, test,eval_train,
#                                                                epochs, k=atk)
#        curves.append({'params': params,
#                       'patk': {'train': train_patk, 'test': test_patk},
#                       'mse': {'train': train_mse, 'test': test_mse}})
#    return curves



#%% 
# =============================================================================
#  load data from sql server
# =============================================================================

#con = pypyodbc.connect("DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=xxxxxxx;DATABASE=project2017")

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
# =============================================================================
# train model, no- features used ( naive test ----)
# =============================================================================
NUM_THREADS =4
NUM_COMPONENTS = 30

eval_train = train.copy()
non_eval_users = list(set(range(train.shape[0])) - set(user_idxs)) ## 

eval_train = eval_train.tolil()
for u in non_eval_users:
    eval_train[u, :] = 0.0
eval_train = eval_train.tocsr()

model = LightFM(loss='warp',
                no_components=NUM_COMPONENTS,
                random_state=2017)
model.fit(train,
          epochs=20,
          num_threads=4,verbose=True)


# sanity check
# Don't forget the pass in the item features again!
train_auc = auc_score(model,
                      eval_train,                      
                      num_threads=NUM_THREADS).mean()
train_recall = recall_at_k(model,
                           eval_train,
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
print('*****TEST*****\n\tauc : {0:.2f}\n'.format(test_auc))
print('\t recall: {0:.2f}'.format(test_recall))

#%% 
model = LightFM(loss='warp',
                no_components=NUM_COMPONENTS,
                random_state=2017)

model_k10, train_p10, test_p10,train_r10, test_r10, train_rr10, test_rr10 = \
learning_curve(model,train,test,eval_train,iterarray=[1,10,15,20],k=10)


#%% 

# =============================================================================
# optimizing Hyperparameter with scikit-optimize (linux only)
# =============================================================================
from skopt import forest_minimize

space = [(1, 100), # epochs
         (10**-4, 1.0, 'log-uniform'), # learning_rate
         (20, 200), # no_components
         (10**-6, 10**-1, 'log-uniform'), # alpha
        ]

res_fm = forest_minimize(objective, space, 
                         n_calls=250,
                         random_state=0,
                         verbose=True)

print('Maximimum p@k found: {:6.5f}'.format(-res_fm.fun))
print('Optimal parameters:')
params = ['epochs', 'learning_rate', 'no_components', 'alpha']
for (p, x_) in zip(params, res_fm.x):
    print('{}: {}'.format(p, x_))
""" optimize params 
Maximimum p@k found: 0.03424
Optimal parameters:
epochs: 98
learning_rate: 0.059126434145345366
no_components: 31
alpha: 0.0006973742463233035    
"""
## save varaibale(res_fm) 
## -- train take several hours in my laptop
with open('./funds/res_fm_no_feat.pickle',mode='wb') as f:
    pickle.dump(res_fm,f)
    
#%%

