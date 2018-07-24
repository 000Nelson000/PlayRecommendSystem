#! encoding : utf8
import pandas as pd 
import numpy as np 
from collections import Counter,OrderedDict

def _getSet(df,col,delimiter):
    s = Counter()
    np_array = []
    for _,e in enumerate(df[col].values):
        row = []
        for e1 in e.split(delimiter):
            # s.add(e1)
            row.append(e1)
            s[e1] += 1 
        np_array.append(row)
    return list(s.keys()),np_array

def _arrange_to_dummy(feat_union,np_lst_array):    
    num_rows = len(np_lst_array)
    num_cols = len(feat_union)
    data = np.zeros(shape=(num_rows,num_cols),dtype='uint8')
    feat_union = np.array(feat_union)
    for ridx,row in enumerate(np_lst_array):
        for e in row:
            cidx = np.where(feat_union==e)[0][0]
            data[ridx,cidx] = 1
    
    return data

def convert_to_dummy_df(df,col,delimiter=','):    
    feat_sets, np_feat_array = _getSet(df,col,delimiter)
    np_dummy = _arrange_to_dummy(feat_sets,np_feat_array)
    feat_colnames = [col + '_' + feat for feat in feat_sets]
    dummies_df = pd.DataFrame(np_dummy)
    dummies_df.columns = feat_colnames    
    ## reorder column by names
    df = dummies_df.reindex_axis(sorted(dummies_df.columns), axis=1)
    return df
#%%
if __name__ == '__main__':
    df = pd.DataFrame({
        'feat':[
            'a,b,c,d',
            '1,3,5',
            'e,f,a'
        ]
    })
    print(df)
    # print(getSet(df,'feat'))
    df_dummy = convert_to_dummy_df(df,'feat')
    print(df_dummy)

    
