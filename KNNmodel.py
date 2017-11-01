# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:41:13 2017

@author: 116952
"""
import numpy as np 
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from rec_helper import *
# =============================================================================
# build funds recommendation 
# =============================================================================
class KNNmodel:
    
    def __init__(self,interaction,kind='ubcf'):

        assert kind.lower() in ('ubcf','ibcf','popular','ubcf_fs')
        self.inter = interaction # u-i interaction data, should be user in rows, items in columns
        self.kind = kind # which kind of algo
        self.sim = None # similarity sparse matrix
        self.rating = None # rating (score) for recommended items
        self.topK = None # top K nearest neighbors
        self.topN = None # top N items for recommendation
        self.precision = None 
        self.recall = None 

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
#        similarities = similarities.tocoo()
        similarities.setdiag(0)
        similarities = similarities.tocsr()
        sparsity = float(similarities.nnz / mat.shape[0]**2) * 100
        print('similarity (jaccard) matrix built ({}), \nsparsity of similarity: {:.2f} %'\
              .format(self.kind,sparsity))
        self.sim = similarities

    def cosine_sim(self):
        """calculate cosine similarity based on (ubcf/ibcf) and store it in self.sim"""
        if self.kind == 'ubcf':
            self.sim = cosine_similarity(self.inter,dense_output=False)
            self.sim.setdiag(0)
            sparsity = float(self.sim.nnz / self.inter.shape[0]**2) * 100
            print('similarity (cosine) matrix build (ubcf), \nsparsity of similarity: {:.2f} %'\
                  .format(sparsity))
        elif self.kind =='ibcf':
            self.sim = cosine_similarity(self.inter.T,dense_output=False)
            self.sim.setdiag(0)
            sparsity = float(self.sim.nnz / self.inter.shape[1]**2) * 100
            print('similarity (cosine) matrix build (ibcf), \nsparsity of similarity: {:.2f} %'\
                  .format(sparsity))
    
    def _replace_purcashed_items(self,X):
        """replace items which user have already bought
        """
        X_coo = sp.coo_matrix(X)
        row = X_coo.row
        col = X_coo.col
        data = X_coo.data
        idx = 0
        inter_coo = self.inter.tocoo()
        ij = zip(inter_coo.row,inter_coo.col)
        intersect_ij = set(ij)

        for i,j,v in tqdm(zip(X_coo.row,X_coo.col,X_coo.data),
                          total = len(X_coo.data)):
            if (i,j) in intersect_ij:
                data[idx] = 0
            idx+=1
        X_coo = sp.coo_matrix((data,(row,col)))

        return X_coo.tocsr()

    def fit(self,topK=100,normalize=True,remove=False,user_features=None):
        """fit model

        params
        ======
        topK: (int)
            consider K nearest neighbors
        normalize: (True/False)
            normalize similarity w.r.t item/user
        remove: (True/False)
            remove rating(scoring) which user bought

        attributes
        ==========
        self.rating: sparse matrix(csr) (num_users*num_items)
            rating/scoring for every items for every users
        self.topK: (int)
            nearest neighbors
        """
        pred = sp.lil_matrix((self.inter.shape))
        rating = self.inter


        if self.kind == 'ubcf':
            sim = self.sim
            ## memory cost a lot if num_of users is very large
#            topK_users = np.argsort(sim.A,axis=1)[:,:-topK-1:-1]  
            ##### build topK_users array ####
            topK_users = np.zeros(shape=(sim.shape[0],topK))
            print('-- start building topK user array...')
            for row_idx in tqdm(range(sim.shape[0]),unit='users'):
                topk_users = np.argsort(sim[row_idx,].A,)[:,:-topK-1:-1]
                topK_users[row_idx,] = topk_users
            ## 
#            for row_idx in tqdm(range(sim.shape[0]),unit='user'):
#                slice_start = sim.indptr[row_idx]
#                slice_end = sim.indptr[row_idx+1]
#                sim_row = sim.data[slice_start:slice_end] 
#                col_idx_by_row = sim.indices[slice_start:slice_end]
#                zipped_list = list(zip(col_idx_by_row,sim_row))
#                zipped_list.sort(key=lambda e:e[1],reverse=True)
#                sorted_col_idx = [idx[0] for idx in zipped_list][:topK]
#                ## pading with different len
#                sorted_col_idx = np.pad(sorted_col_idx,(0,topK - len(sorted_col_idx)),
#                       'constant',constant_values = 0)
#                topK_users[row_idx,] = sorted_col_idx 
            print('-- end building topK user array---')
            
                
               
            print('start building prediction rating...')        
            for user in tqdm(range(pred.shape[0]),unit='users'):
                topk_user = topK_users[user,:] # shape:(topK,)
                #top k users for a given user , but pretty slow if numofusers large
#                topk_user = np.argsort(sim[:,user].A)[0,:-topK-1:-1]
                pred[user,:] = sim[user,topk_user].dot(\
                    rating[topk_user,:])
#                pred[user,:] /= np.sum(np.abs(sim[user,:])) # extremely slow

            if normalize:
                eps = 1e-5
                denominator = sim.sum(axis=1)
                zero_idx = np.where(np.isclose(denominator,0)) # which is zero in denominator is not allowd
                denominator[zero_idx[0],] = np.full((len(zero_idx[0]),1),eps)
                pred /= denominator


        elif self.kind =='ibcf':
            sim = self.sim
#            topK_items = np.argsort(sim.A,axis=1)[:,:-topK-1:-1] #memory cost a lot
            for item in tqdm(range(pred.shape[1]),unit='items'):
                #top k items for a give item
                topk_item = np.argsort(sim[item,].A)[0,:-topK-1:-1]
                #
#                topk_item = topK_items[item,:] # shape: (topK,)
                pred[:,item] = rating[:,topk_item].dot(\
                    sim[topk_item,item])
#                pred[:,item] /= np.sum(np.abs(sim[:,item])) # extremely slow

            if normalize:
#                np.seterr(divide='ignore',invalid='ignore') # suppress warning message
                eps = 1e-5
                denominator = sim.sum(axis=0)
                zero_idx = np.where(np.isclose(denominator,0)) # which is zero
                denominator[zero_idx] = np.full((1,len(zero_idx[0])),eps)
                pred /= denominator

        elif self.kind =='popular':
            num_of_sell_per_item = np.array(self.inter.sum(axis=0).tolist()[0]) # np-array
            topK_pop_indices = np.argsort(num_of_sell_per_item)[:-topK-1:-1]
            topK_num_of_pop =  num_of_sell_per_item[topK_pop_indices]
            if normalize:
                topK_num_of_pop = topK_num_of_pop/ np.sum(topK_num_of_pop)

            for user in tqdm(range(pred.shape[0])):
                pred[user,topK_pop_indices] = topK_num_of_pop

        elif self.kind == 'ubcf_fs':
            """built CF model based on -- feature selected ubcf """
            assert user_features.shape[0] == self.inter.shape[0]

            inter_copy = self.inter.copy() # copy of interaction data
            user_features = user_features.A
            for idx in range(user_features.shape[1]):                
                inter_temp = inter_copy.copy()
                rows_zeros = np.where(user_features[:,idx])[0]
                csr_rows_set_nz_to_val(inter_temp,rows_zeros, value=0) ## set to zero
                model = KNNmodel(inter_temp,kind='ubcf')
                model.jaccard_sim()
                model.fit(topK=100)

                pred+= model.rating


        print('{} rating matrix built...'.format(self.kind))
        if remove:
            print('\nhandling nan data...')
            pred = self._replace_purcashed_items(pred)
        if not remove:
            pred = sp.csr_matrix(pred)
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
        if self.kind in ('ibcf','ubcf','popular','ubcf_fs'):
            topNarray = np.argsort(self.rating[uids,:].A,kind='heapsort')[:,:-topN-1:-1]
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
#            return np.sum(prec_array)/num_of_test_data
            self.precision = np.sum(prec_array)/ num_of_test_data
            print("\n-------------")
            print("model: {},\ntopN: {}".format(self.kind, self.topN))
            print("precision: {:.2f} %".format(self.precision * 100))


        if method == 'recall':
            test_coo = test.tocoo()
            score = 0
            nonzero_rowsets = set(test_coo.row)
            for row,col,v in zip(test_coo.row,test_coo.col,test_coo.data):
                if col in pred_all[row,]:
                    score += 1
            self.recall = score/len(nonzero_rowsets)
            print("\n-------------")
            print("model: {},\ntopN: {}".format(self.kind, self.topN))
            print("recall:{:.2f} %".format(score/len(test_coo.data) * 100))
