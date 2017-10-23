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
        
    
    def fit(self,topK=20):
                
        pred = sp.lil_matrix((self.inter.shape))
        rating = self.inter.tolil()
        sim = self.sim.tolil()
        
        if self.kind == 'ubcf':
            top_K_users = np.argsort(sim.A,axis=1)[:,:-topK-1:-1] 
            for user in range(top_K_users.shape[0]):
                pred[user, ] = sim[user,top_K_users[user]]\
                    .dot(rating[top_K_users[user,:]]) # lil
                pred[user,] /= np.sum(np.abs(sim[user,]))
            self.rating = pred
            
        elif self.kind =='ibcf':
            top_K_items = np.argsort(sim.A,axis=0)[:,:-topK-1:-1]
            for item in range(top_K_items.shape[0]):
                pred[:,item] = rating[:,top_K_items[item,:]].dot(sim[top_K_items[item], item])
                pred[:,item] /= np.sum(np.abs(sim[:,item]))
            self.rating = pred
            
    def predict(self):
        pass
    
    def popular_items(self):
        pass
    def evaluate(self):
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
    idx_to_userid, itemid_to_idx,idx_to_itemid = df_to_spmatrix(df_gt2,'身分證字號','基金代碼')
    train,test, user_idxs = train_test_split(purchased_ui,split_count=1,fraction=0.2)
