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
        
        assert kind.lower() in ('ubcf','ibcf')        
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
        
    def _replace_nan_in_csr(self,X):
        X_coo = sp.coo_matrix(X)
        row = X_coo.row
        col = X_coo.col
        data = X_coo.data
        idx = 0
        for i,j,v in zip(X_coo.row,X_coo.col,X_coo.data):            
            if (np.isnan(v)):
                data[idx] = 0
            idx+=1
        X_coo = sp.coo_matrix((data,(row,col)))
        return X_coo.tocsr()
    
    def fit(self,topK=20,normalize=True):
                
        pred = sp.lil_matrix((self.inter.shape))
        rating = self.inter
        sim = self.sim
        widgets=[Percentage(),Bar()] # progress bar 
        
        if self.kind == 'ubcf':
            pbar = ProgressBar(widgets=widgets,maxval=pred.shape[0]).start()
            topK_users = np.argsort(sim.A,axis=1)[:,:-topK-1:-1]  ## memory cost a lot if users
            for user in range(pred.shape[0]):
                topk_user = topK_users[user,:] # shape:(topK,)
                pred[user,:] = sim[user,topk_user].dot(\
                    rating[topk_user,:])
#                pred[user,:] /= np.sum(np.abs(sim[user,:])) # extremely slow
                pbar.update(user+1)
            if normalize:
                np.seterr(divide='ignore',invalid='ignore') # suppress warning message 
                pred /= sim.sum(axis=1)
                                            
            
        elif self.kind =='ibcf':
            topK_items = np.argsort(sim.A,axis=1)[:,:-topK-1:-1] #            
            
            pbar = ProgressBar(widgets=widgets,maxval=pred.shape[1]).start()
            for item in range(pred.shape[1]):
                topk_item = topK_items[item,:] # shape: (topK,)
                pred[:,item] = rating[:,topk_item].dot(\
                    sim[topk_item,item])
#                pred[:,item] /= np.sum(np.abs(sim[:,item])) # extremely slow
                pbar.update(item+1)
            
            if normalize:
#                print(repr(pred))
                np.seterr(divide='ignore',invalid='ignore') # suppress warning message 
                pred /= sim.sum(axis=0)
                
                
        print('\nhandling nan data...')
        pred = self._replace_nan_in_csr(pred)
        pbar.finish()
        rows = pred.shape[0]
        cols = pred.shape[1]
        sparsity = float( pred.getnnz()/ (rows*cols) )*100
        print('# of rows : {}'.format(rows))
        print('numbers of cols: {}'.format(cols))
        print('sparsity of rating: {:.2f} %'.format(sparsity))
        print('save into *.rating attribute...')
        pred = sp.csr_matrix(pred)
        self.rating = pred                
        
            
    def predict(self,uids,topN=10):
        """predict topN recommended items for uids 
        
        params
        ======
        uids: (array)
        """
        if self.kind in ('ibcf','ubcf'):
            topNarray = np.argsort(self.rating[uids,:].A)[:,-topN-1:-1]
            return topNarray
        
        
    
    def popular_items(self):
        pass
    def evaluate(self,method = 'precision'):
        pass
# %%    
# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    import pandas as pd
    import numpy as np 
    import scipy.sparse as sp
    from sklearn.metrics.pairwise import cosine_similarity
    from progressbar import ProgressBar, Percentage,Bar
    
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
