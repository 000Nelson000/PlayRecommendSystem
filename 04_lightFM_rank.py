#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 21:23:25 2017

'copy' from : data pique | Learning to rank
http://blog.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/
"""

#%% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit
import pickle
import csv
import copy
import itertools
from lightfm import LightFM
import lightfm.evaluation
import sys
sys.path.append('../')
import helpers

df = pd.read_csv('./rec-a-sketch/model_likes_anon.psv',
                 sep='|', quoting=csv.QUOTE_MINIMAL,
                 quotechar='\\')
df.drop_duplicates(inplace=True)
df.head()

#%% 
# Threshold data to only include users and models with min 5 likes.
df = helpers.threshold_interactions_df(df, 'uid', 'mid', 5, 5)

# Go from dataframe to likes matrix
# Also, build index to ID mappers.

likes, uid_to_idx, idx_to_uid,\
mid_to_idx, idx_to_mid = helpers.df_to_matrix(df, 'uid', 'mid')

#likes 
train, test, user_index = helpers.train_test_split(likes, 5, fraction=0.2)
# remove 
eval_train = train.copy()
non_eval_users = list(set(range(train.shape[0])) - set(user_index)) # set diff

eval_train = eval_train.tolil()
for u in non_eval_users:
    eval_train[u, :] = 0.0
eval_train = eval_train.tocsr()

#%%
# =============================================================================
# Encode side-information,(item/users' features)
# =============================================================================
sideinfo = pd.read_csv('./rec-a-sketch/model_feats.psv',
                       sep='|')

# Build list of dictionaries containing features 
# and weights in same order as idx_to_mid prescribes.
feat_dlist = [{} for _ in idx_to_mid]
for idx, row in sideinfo.iterrows():
#    print(idx,row)
    feat_key = '{}_{}'.format(row.type, str(row.value).lower())
    idx = mid_to_idx.get(row.mid)
    if idx is not None:
        feat_dlist[idx][feat_key] = 1

feat_dlist[0] # features dictionary 

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
item_features = dv.fit_transform(feat_dlist)
item_features # 25655x20352

#%% 
# =============================================================================
# training
# =============================================================================

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

def patk_learning_curve(model, train, test, eval_train,
                        iterarray, user_features=None,
                        item_features=None, k=5,
                        **fit_params):
    old_epoch = 0
    train_patk = []
    test_patk = []
    headers = ['Epoch', 'train p@5', 'test p@5']
    print_log(headers, header=True)
    for epoch in iterarray:
        more = epoch - old_epoch
        model.fit_partial(train, user_features=user_features,
                          item_features=item_features,
                          epochs=more, **fit_params)
        this_test = lightfm.evaluation.precision_at_k(model, test, train_interactions=None, k=k)
        this_train = lightfm.evaluation.precision_at_k(model, eval_train, train_interactions=None, k=k)

        train_patk.append(np.mean(this_train))
        test_patk.append(np.mean(this_test))
        row = [epoch, train_patk[-1], test_patk[-1]]
        print_log(row)
    return model, train_patk, test_patk

#%% 
# =============================================================================
#     training 
# =============================================================================

model = lightfm.LightFM(loss='warp', random_state=2016)
# Initialize model.
model.fit(train, epochs=0);

iterarray = range(10, 110, 10)

model, train_patk, test_patk = patk_learning_curve(
    model, train, test, eval_train, iterarray, k=5, **{'num_threads': 4}
)

#%%
# =============================================================================
# plot
# =============================================================================
import seaborn as sns
sns.set_style('white')

def plot_patk(iterarray, patk,
              title, k=5):
    plt.plot(iterarray, patk);
    plt.title(title, fontsize=20);
    plt.xlabel('Epochs', fontsize=24);
    plt.ylabel('p@{}'.format(k), fontsize=24);
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);

# Plot train on left
ax = plt.subplot(1, 2, 1)
fig = ax.get_figure();
sns.despine(fig);
plot_patk(iterarray, train_patk,
         'Train', k=5)

# Plot test on right
ax = plt.subplot(1, 2, 2)
fig = ax.get_figure();
sns.despine(fig);
plot_patk(iterarray, test_patk,
         'Test', k=5)

plt.tight_layout();

#%%

# =============================================================================
# Learning to Rank + Side information 
# =============================================================================

# Need to hstack item_features
eye = sp.eye(item_features.shape[0], item_features.shape[0]).tocsr()
item_features_concat = sp.hstack((eye, item_features))
item_features_concat = item_features_concat.tocsr().astype(np.float32)

## New Object function
def objective_wsideinfo(params):
    # unpack
    epochs, learning_rate,\
    no_components, item_alpha,\
    scale = params
    
    user_alpha = item_alpha * scale
    model = lightfm.LightFM(loss='warp',
                    random_state=2016,
                    learning_rate=learning_rate,
                    no_components=no_components,
                    user_alpha=user_alpha,
                    item_alpha=item_alpha)
    model.fit(train, epochs=epochs,
              item_features=item_features_concat,
              num_threads=4, verbose=True)
    
    patks = lightfm.evaluation.precision_at_k(model, test,
                                              item_features=item_features_concat,
                                              train_interactions=None,
                                              k=5, num_threads=3)
    mapatk = np.mean(patks)
    # Make negative because we want to _minimize_ objective
    out = -mapatk
    # Weird shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out

#%% 
# =============================================================================
# Fun with features embedding
# =============================================================================
## optimial params ###        
#        Maximimum p@k found: 0.04610
#Optimal parameters:
epochs =  192
learning_rate= 0.06676184785227865
no_components= 86
item_alpha =  0.0005563892936299544
scale =  0.6960826359109953
learning_rate = 0.06676184785227865 ## 
user_alpha = item_alpha * scale


model = lightfm.LightFM(loss='warp', 
                        random_state=2016,
                        learning_rate = learning_rate,
                        no_components = no_components,
                        user_alpha = user_alpha,
                        item_alpha = item_alpha)
model.fit(train, epochs = epochs,
          item_features= item_features_concat,num_threads=4,verbose=True)

# %% 
idx = dv.vocabulary_['tag_tiltbrush'] + item_features.shape[0]
def cosine_similarity(vec, mat):
    sim = vec.dot(mat.T)
    matnorm = np.linalg.norm(mat, axis=1)
    vecnorm = np.linalg.norm(vec)
    return np.squeeze(sim / matnorm / vecnorm)

tilt_vec = model.item_embeddings[[idx], :]
item_representations = item_features_concat.dot(model.item_embeddings)
sims = cosine_similarity(tilt_vec, item_representations)

import requests
def get_thumbnails(row, idx_to_mid, N=10):
    thumbs = []
    mids = []
    for x in np.argsort(-row)[:N]:
        response = requests.get('https://sketchfab.com/i/models/{}'\
                                .format(idx_to_mid[x])).json()
        thumb = [x['url'] for x in response['thumbnails']['images']
                 if x['width'] == 200 and x['height']==200]
        if not thumb:
            print('no thumbnail')
        else:
            thumb = thumb[0]
        thumbs.append(thumb)
        mids.append(idx_to_mid[x])
    return thumbs, mids

from IPython.display import display, HTML

def display_thumbs(thumbs, mids, N=5):
    thumb_html = "<a href='{}' target='_blank'>\
                  <img style='width: 160px; margin: 0px; \
                  float: left; border: 1px solid black;' \
                  src='{}' /></a>"
    images = ''
    for url, mid in zip(thumbs[0:N], mids[0:N]):
        link = 'http://sketchfab.com/models/{}'.format(mid)
        images += thumb_html.format(link, url)
    display(HTML(images))
display_thumbs(*get_thumbnails(sims, idx_to_mid))
#%% 
# =============================================================================
# tag suggestion
# =============================================================================
idx = 900
mid = idx_to_mid[idx]
def display_single(mid):
    """Display thumbnail for a single model"""
    response = requests.get('https://sketchfab.com/i/models/{}'\
                            .format(mid)).json()
    thumb = [x['url'] for x in response['thumbnails']['images']
             if x['width'] == 200 and x['height']==200][0]
    thumb_html = "<a href='{}' target='_blank'>\
                  <img style='width: 200px; margin: 0px; \
                  float: left; border: 1px solid black;' \
                  src='{}' /></a>"
    link = 'http://sketchfab.com/models/{}'.format(mid)
    display(HTML(thumb_html.format(link, thumb)))

display_single(mid)

# Make mapper to map from from feature index to feature name
idx_to_feat = {v: k for (k, v) in dv.vocabulary_.items()}
print('Tags:')
for i in item_features.getrow(idx).indices:
    print('- {}'.format(idx_to_feat[i]))
    

# Indices of all tag vectors
tag_indices = set(v for (k, v) in dv.vocabulary_.items()
                  if k.startswith('tag_'))
# Tags that are already present
filter_tags = set(i for i in item_features.getrow(idx).indices)

item_representation = item_features_concat[idx, :].dot(model.item_embeddings)
sims = cosine_similarity(item_representation, model.item_embeddings)

suggested_tags = []
i = 0
recs = np.argsort(-sims)
n_items = item_features.shape[0]
while len(suggested_tags) < 10:
    offset_idx = recs[i] - n_items
    if offset_idx in tag_indices\
       and offset_idx not in filter_tags:
        suggested_tags.append(idx_to_feat[offset_idx])
    i += 1
print('Suggested Tags:')
for t in suggested_tags:
    print('- {}'.format(t))