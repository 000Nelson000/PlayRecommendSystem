# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:41:13 2017

@author: 116952
"""

#%% 
# =============================================================================
# knn model 
# =============================================================================
#__all__ = ['KNNmodel']

class KNNmodel:
    def __init__(self,interaction,kind='ubcf'):
        
        assert kind.lower() in ('ubcf','ibcf','popular')
        self.inter = interaction
        self.kind = kind
        
    
    def jaccard_sim(self):
        '''given a sparse matrix, calculate jaccard sim         
        
        ** ref : http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
        '''
        if self.kind == 'ubcf':
            mat = self.inter
        elif self.kind =='ibcf':
            mat = self.inter.T
                    
        rows_sum = mat.getnnz(axis=1)  # 
        ab = mat.dot(mat.T) # mat x t(mat)
        ab = ab.astype('float32')        
        # for rows
        aa = np.repeat(rows_sum, ab.getnnz(axis=1))
        # for columns
        bb = rows_sum[ab.indices]
    
        similarities = ab.copy()
        similarities.data /= (aa + bb - ab.data)
        similarities.setdiag(0)
        similarities = similarities.tocsr()
        sparsity = float(similarities.nnz / mat.shape[0]**2) * 100
        print('similarity matrix built ({}), sparsity: {:.2f} %'\
              .format(self.kind,sparsity))
        self.sim = similarities
        
    def cosine_sim(self):
        """calculate cosine similarity based on (ubcf/ibcf) and store it in self.sim"""        
        if self.kind == 'ubcf':
            self.sim = cosine_similarity(self.inter,dense_output=False)
            self.sim.setdiag(0)
            sparsity = float(self.sim.nnz / self.inter.shape[0]**2) * 100
            print('similarity matrix build (ubcf), sparsity: {:.2f} %'\
                  .format(sparsity))            
        elif self.kind =='ibcf':
            self.sim = cosine_similarity(self.inter.T,dense_output=False)
            self.sim.setdiag(0)
            sparsity = float(self.sim.nnz / self.inter.shape[1]**2) * 100            
            print('similarity matrix build (ibcf), sparsity: {:.2f} %'\
                  .format(sparsity))
        
    def _replace_nan_in_csr(self,X,remove):
        X_coo = sp.coo_matrix(X)
        row = X_coo.row
        col = X_coo.col
        data = X_coo.data
        idx = 0
        inter_coo = self.inter.tocoo()
        ij = zip(inter_coo.row,inter_coo.col)
        intersect_ij = set(ij)
        for i,j,v in tqdm(zip(X_coo.row,X_coo.col,X_coo.data)):
            if (np.isnan(v)):
                data[idx] = 0
            if remove:
                if (i,j) in intersect_ij:
                    data[idx] = 0
            idx+=1
        X_coo = sp.coo_matrix((data,(row,col)))
        
#        inter_lil = self.inter.tolil() # interaction datasets (csr -> lil)
#        inter_lil_rows = inter_lil.rows
#        
#        X_lil = sp.lil_matrix(X)
#        
#        for r_idx, row in enumerate(X_lil.rows):
#            remove_intersec = np.intersect1d(row,inter_lil_rows[r_idx])
#            for col_idx in row:
#                if np.isnan(X_lil[r_idx,col_idx]):
#                    X_lil[r_idx,col_idx] = 0
#            if remove:# remove existed rating data
#                for remove_cidx in remove_intersec:
#                    X_lil[r_idx,remove_cidx] = 0
                    
        return X_coo.tocsr()
    
    def fit(self,topK=20,normalize=True,remove=False):
        """fit model
        
        params
        ======
        topK: (int)
            consider K nearest neighbors 
        normalize: (True/False)
            normalize similarity w.r.t item/user
        remove: (True/False)
            remove rating(scoring) in purchased items 
        
        attributes
        ==========
        self.rating: sparse matrix(csr) (num_users*num_items)
            rating/scoring for every items for every users 
        self.topK: (int)
            nearest neighbors
        """
        pred = sp.lil_matrix((self.inter.shape))
        rating = self.inter
        
#        widgets=[Percentage(),Bar()] # progress bar 
        
        if self.kind == 'ubcf':
            sim = self.sim
#            pbar = ProgressBar(widgets=widgets,maxval=pred.shape[0]).start()
            topK_users = np.argsort(sim.A,axis=1)[:,:-topK-1:-1]  ## memory cost a lot if num_of users is very large
            for user in tqdm(range(pred.shape[0])):
                topk_user = topK_users[user,:] # shape:(topK,)
                pred[user,:] = sim[user,topk_user].dot(\
                    rating[topk_user,:])
#                pred[user,:] /= np.sum(np.abs(sim[user,:])) # extremely slow
#                pbar.update(user+1)
            if normalize:
                np.seterr(divide='ignore',invalid='ignore') # suppress warning message 
                pred /= sim.sum(axis=1)
                                            
            
        elif self.kind =='ibcf':
            sim = self.sim
            topK_items = np.argsort(sim.A,axis=1)[:,:-topK-1:-1] #memory cost a lot            
#            pbar = ProgressBar(widgets=widgets,maxval=pred.shape[1]).start()
            for item in tqdm(range(pred.shape[1])):
                topk_item = topK_items[item,:] # shape: (topK,)
                pred[:,item] = rating[:,topk_item].dot(\
                    sim[topk_item,item])
#                pred[:,item] /= np.sum(np.abs(sim[:,item])) # extremely slow
#                pbar.update(item+1)
            
            if normalize:
#                print(repr(pred))
                np.seterr(divide='ignore',invalid='ignore') # suppress warning message 
                pred /= sim.sum(axis=0)
                
        elif self.kind =='popular':
#            pbar = ProgressBar(widgets=widgets,maxval=pred.shape[0]).start()
            num_of_sell_per_item = np.array(self.inter.sum(axis=0).tolist()[0]) # np-array
            topK_pop_indices = np.argsort(num_of_sell_per_item)[:-topK-1:-1]
            topK_num_of_pop =  num_of_sell_per_item[topK_pop_indices]
            if normalize:
                topK_num_of_pop = topK_num_of_pop/ np.sum(topK_num_of_pop)
            
            for user in tqdm(range(pred.shape[0])):
                pred[user,topK_pop_indices] = topK_num_of_pop
#                pbar.update(user+1)
            
        
#        pbar.finish()        
        print('{} rating matrix built...'.format(self.kind))
        print('\nhandling nan data...')
        pred = self._replace_nan_in_csr(pred,remove=remove)
        
        rows = pred.shape[0]
        cols = pred.shape[1]
        sparsity = float( pred.getnnz()/ (rows*cols) )*100
        print('numbers of rows : {}'.format(rows))
        print('numbers of cols: {}'.format(cols))
        print('sparsity of rating: {:.2f} %'.format(sparsity))
        print('save into *.rating attribute...')
        pred = sp.csr_matrix(pred)
        self.rating = pred
        self.topK = topK
        
            
    def predict(self,uids,topN=10):
        """predict topN recommended items for uids 
        
        params
        ======
        uids: (array)
        topN: (int)
        
        attributes
        ==========
        topN: (int)
            top N recommend items 
        
        returns
        =======
        topNarray: (2darray)
            top N recommeded items idx array for uids
        """
        if self.kind in ('ibcf','ubcf','popular'):
            topNarray = np.argsort(self.rating[uids,:].A)[:,:-topN-1:-1]
            self.topN = topN
            return topNarray
        
        
    
    def evaluate(self,pred_all,test,method = 'precision'):
        """
        params
        ======
        pred_all:
        test:
        pred:
        method: (str) precision/recall/...
        
        attribute
        =========
        precision
        recall
        
        """
        assert type(test) == sp.csr_matrix 
        assert test.shape[0] == pred_all.shape[0]
        
        if method == 'precision':            
            test_lil = test.tolil()
            prec_array = np.zeros(pred_all.shape[0])
            num_of_test_data = 0
            for user,items in enumerate(test_lil.rows):
                prec_array[user] = len(np.intersect1d(items,pred_all[user,])) / len(pred_all[user,])
                if items != []:
                    num_of_test_data += 1
            return np.sum(prec_array)/num_of_test_data
                    
                
        if method == 'recall':
            test_coo = test.tocoo()
            score = 0
            nonzero_rowsets = set(test_coo.row)
            for row,col,v in zip(test_coo.row,test_coo.col,test_coo.data):    
                if col in pred_all[row,]:
                    score += 1
            self.recall = score/len(nonzero_rowsets)
            print("\n-------------")
            print("model: {}, topN: {}".format(self.kind, self.topN))
            print("recall:{:.2f} %".format(score/len(test_coo.data) * 100))
        
            
            
            
# %%    
# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    import pandas as pd
    import numpy as np 
    import scipy.sparse as sp
    from sklearn.metrics.pairwise import cosine_similarity
#    from progressbar import ProgressBar, Percentage,Bar
    from tqdm import tqdm
    
    import rec_helper
    
    df_inter = pd.read_csv('./funds/purchase.csv',encoding='cp950')
    df_item = pd.read_csv('./funds/item_features.csv',encoding='cp950')
    df_user = pd.read_csv('./funds/user_features.csv',encoding='cp950')

    #%%
    ### there are some fundids in df_inter not exists in df_item 
    fundids_df_items = df_item['基金代碼'].as_matrix() # 1d array
    fundids_df_inter = df_inter['基金代碼'].unique() # 1d array
    fundids = np.intersect1d(fundids_df_inter,fundids_df_items) # 1d array
    
    ### arrange purchasing data which fundid exist in fundids
    ## (exclude data which is not exist in fundids)
    df_inter = df_inter.loc[df_inter['基金代碼'].isin(fundids)]
    ## user who bought at least two items
    df_gt2 = rec_helper.threshold_interaction(df_inter,'身分證字號','基金代碼') # 
    ### 
#    purchased_ui1, userid_to_idx1, \
#    idx_to_userid1, itemid_to_idx1,idx_to_itemid1= df_to_spmatrix(df_inter,'身分證字號','基金代碼')
#    
    #train,test, user_idxs = train_test_split(purchased_ui,split_count=1,fraction=0.2)
    
    
    purchased_ui, userid_to_idx, \
    idx_to_userid, itemid_to_idx,idx_to_itemid = rec_helper.df_to_spmatrix(df_gt2,'身分證字號','基金代碼')
    train,test, user_idxs = rec_helper.train_test_split(purchased_ui,split_count=1,fraction=0.2)
    #%%
    # =============================================================================
    #  model    
    # =============================================================================
    ## ibcf
    model_i = KNNmodel(train,kind='ibcf')
    model_i.jaccard_sim()
    model_i.fit(topK=100,remove=True)
    ## ubcf
    model_u = KNNmodel(train,kind='ubcf')
    model_u.jaccard_sim()
    model_u.fit(topK=100,remove=True)
    ## popular 
    model_p = KNNmodel(train,kind='popular')
    model_p.fit(topK=100,remove=True)


    #%%
    # =============================================================================
    # evaluate recall
    # =============================================================================
    uids = np.arange(0,train.shape[0])
    
    predall_u = model_u.predict(uids,topN=10)
    model_u.evaluate(predall_u,test,method='recall') # 28.4 %
    
    predall_i = model_i.predict(uids,topN=10)
    model_i.evaluate(predall_i,test,method='recall') # 12.85 %

    predall_p = model_p.predict(uids,topN=10)
    model_p.evaluate(predall_p,test,method='recall') # 20.57 %
    
    # %%
    temp_p = model_p.evaluate(predall_p,test,method='precision')
#%%
    
# =============================================================================
# test    
# =============================================================================
#model = model_p 
#
#def test_eval(model):
#    uids = np.arange(0,train.shape[0])
#    pred_all = model.predict(uids,5) # 2d array
#    test_coo = test.tocoo()
#    score = 0
#    for row,col,v in zip(test_coo.row,test_coo.col,test_coo.data):    
#        if col in pred_all[row,]:
#            score += 1
#    print("\n-------------")
#    print("model: {}".format(model.kind))
#    print("recall:{:.2f} %".format(score/len(test_coo.data) * 100))
#
#models = [model_i,model_u,model_p]
#
#for m in models:
#    test_eval(m)
#    