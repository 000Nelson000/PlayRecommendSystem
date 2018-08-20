# -*- coding: utf-8 -*-
"""

Created on Mon Nov  6 15:26:51 2017

@author: 116952
"""
#%% 
# =============================================================================
# lightfm 
# =============================================================================

from scipy import sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k,recall_at_k,auc_score
import copy
import itertools
import numpy as np 
import lightfm
import pickle 
import pandas as pd 
import time 
#fundid_names_df.to_csv('./funds/fundid_to_name.csv',index=False)

with open('./funds-dataset/sp_funds_datasets.pickle','rb') as f:
    data = pickle.load(f)
    
test = data['test']
train = data['train']
user_idxs = data['user_idxs']
idx_to_userid = data['idx_to_userid']
userid_to_idx = data['userid_to_idx']
idx_to_itemid = data['idx_to_itemid']
itemid_to_idx = data['itemid_to_idx']

fundid_names_df = pd.read_csv('./funds-dataset/fundid_to_name.csv',encoding='cp950')
fundid_to_names = {}

for d in fundid_names_df.to_dict('records'):
    fundid_to_names[d['基金代碼']] = d['基金中文名稱']
#%% 
t1 = time.time()
model_lr = LightFM(learning_rate=0.01, loss='warp')
model_lr.fit(train, epochs=10)
t2 = time.time()
print('model built (lightfm) cost :{:.1f} s'.format(t2-t1))
train_precision = precision_at_k(model_lr, train, k=10).mean()
test_precision = precision_at_k(model_lr, test, k=10).mean()
train_recall = recall_at_k(model_lr,train,k=10).mean()
test_recall = recall_at_k(model_lr,test,k=10).mean()

train_auc = auc_score(model_lr, train).mean()
test_auc = auc_score(model_lr, test).mean()
## on test : Recall- 19.30%, Precision- 1.93%, (AUC-0.91)
print('Recall: train {:.2f}%, test {:.2f}%'.format(100*train_recall,100*test_recall)) 
print('Precision: train {:.2f}% , test {:.2f}%.'.format(100*train_precision, 100*test_precision))
print('AUC: train {:.2f}, test {:.2f}.'.format(train_auc, test_auc))


#%% search 
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
#%%
eval_train = train.copy()    
non_eval_users = list(set(range(train.shape[0])) - set(user_idxs)) ## 
eval_train = eval_train.tolil()
for u in non_eval_users:
    eval_train[u,:] = 0.0
eval_train = eval_train.tocsr()

model_k10, train_p10, test_p10,train_r10, test_r10, train_rr10, test_rr10 = \
learning_curve(model,train,test,eval_train,iterarray=[1,10,50,100],k=10)


#%% grid search 
def grid_search_learning_curve(base_model, train, test, eval_train,param_grid,epochs,
                               atk=10):        
    """grid search 
    
    "Inspired" (stolen) from sklearn gridsearch
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
    
    """
    curves = []
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        this_model = copy.deepcopy(base_model)
        print_line = []
        for k, v in params.items():
            setattr(this_model, k, v)
            print_line.append((k, v))

        print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
        _, train_patk, test_patk, train_ratk, test_ratk,train_rratk,test_rratk = \
        learning_curve(this_model, train, test,eval_train,epochs, k=atk)
        
        curves.append({'params': params,
                       'patk': {'train': train_patk, 'test': test_patk},
                       'ratk': {'train': train_ratk, 'test': test_ratk},
                       'rratk':{'train': train_rratk, 'test':test_rratk}
                       })
    return curves

grid = {
            'loss':['bpr','warp'],
            'learning_rate':[1,0.1,0.05,0.01]
        }
        
curves = grid_search_learning_curve(model,
                                    train,
                                    test,
                                    eval_train,
                                    grid,
                                    epochs=[1,10,50,100],
                                    atk=10)
#%% 
# =============================================================================
# sample recommendation 
# =============================================================================
def sample_recommendation(model, data, user_ids, print_output=True):
    
    train = data['train']
    test = data['test']
    assert isinstance(train,sp.csr_matrix) and isinstance(test,sp.csr_matrix)
        
    n_users, n_items = train.shape

    for user_id in user_ids:
        
        known_positives_itemids = [ 
                idx_to_itemid[e] for e in train[user_id].indices
                ]
        known_positives_item_names = [
                fundid_to_names[e] for e in known_positives_itemids
                ]
        scores = model.predict(user_id, np.arange(n_items))
        top_items_ids = [idx_to_itemid[e] for e in np.argsort(-scores)]
        if print_output == True:
            print("User %s" % user_id)
            print("     Known positives:")

            for x in known_positives_item_names[:3]:
                print("        %s" % x)

            print("     Recommended:")

            for x in top_items_ids[:3]:
                print("        %s" % fundid_to_names[x])
                
sample_recommendation(model,data,range(5))
