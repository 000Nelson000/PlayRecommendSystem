#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Item cold start: recommend stackexchange questions, 

from : lightfm repo

http://lyst.github.io/lightfm/docs/examples/hybrid_crossvalidated.html
"""

# %%

import numpy as np

from lightfm.datasets import fetch_stackexchange

data = fetch_stackexchange('crossvalidated',
                           test_set_fraction=0.1,
                           indicator_features=False,
                           tag_features=True)

train = data['train']
test = data['test']
#%% 
print('The dataset has %s users and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))


#%%
# =============================================================================
# A pure cf model
# =============================================================================
# Import the model
from lightfm import LightFM

# Set the number of threads; you can increase this
# ify you have more physical cores available.
NUM_THREADS = 2
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

# Let's fit a WARP model: these generally have the best performance.
model = LightFM(loss='warp',
                item_alpha=ITEM_ALPHA,
               no_components=NUM_COMPONENTS)

# Run 3 epochs and time it.
%time model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
#%%
# Import the evaluation routines
from lightfm.evaluation import auc_score

# Compute and print the AUC score
train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)

#%%
# sanity check training data --- AUC 

# Import the evaluation routines
from lightfm.evaluation import auc_score

# Compute and print the AUC score
train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)

#%% 
# We pass in the train interactions to exclude them from predictions.
# This is to simulate a recommender system where we do not
# re-recommend things the user has already interacted with in the train
# set.
test_auc = auc_score(model, test, train_interactions=train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)

# %%
"""
The fact that we score them lower than other 
items (AUC < 0.5) is due to estimated per-item biases, 
which can be confirmed by setting them to zero 
and re-evaluating the model.
"""

# Set biases to zero
model.item_biases *= 0.0

test_auc = auc_score(model, test, train_interactions=train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)
#%% 
# =============================================================================
# Hybrid model
# =============================================================================

item_features = data['item_features']
tag_labels = data['item_feature_labels']

print('\nThere are %s distinct tags, with values like %s.' 
      % (item_features.shape[1], tag_labels[:3].tolist()))


#%%
# Define a new model instance
model = LightFM(loss='warp',
                item_alpha=ITEM_ALPHA,
                no_components=NUM_COMPONENTS)

# Fit the hybrid model. Note that this time, we pass
# in the item features matrix.
model = model.fit(train,
                item_features=item_features,
                epochs=NUM_EPOCHS,
                num_threads=NUM_THREADS)

#%%
# sanity check
# Don't forget the pass in the item features again!
train_auc = auc_score(model,
                      train,
                      item_features=item_features,
                      num_threads=NUM_THREADS).mean()
print('Hybrid training set AUC: %s' % train_auc)

#%%
test_auc = auc_score(model,
                    test,
                    train_interactions=train,
                    item_features=item_features,
                    num_threads=NUM_THREADS).mean()
print('Hybrid test set AUC: %s' % test_auc)

#%%

def get_similar_tags(model, tag_id):
    # Define similarity as the cosine of the angle
    # between the tag latent vectors

    # Normalize the vectors to unit length
    tag_embeddings = (model.item_embeddings.T
                      / np.linalg.norm(model.item_embeddings, axis=1)).T

    query_embedding = tag_embeddings[tag_id]
    similarity = np.dot(tag_embeddings, query_embedding)
    most_similar = np.argsort(-similarity)[1:4]

    return most_similar


for tag in (u'bayesian', u'regression', u'survival'):
    tag_id = tag_labels.tolist().index(tag)
    print('Most similar tags for %s: %s' % (tag_labels[tag_id],
                                            tag_labels[get_similar_tags(model, tag_id)]))

#%%
# =============================================================================
#     How to grid-search for all hyper-parameters? 
# =============================================================================
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform as sp_rand

#alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
#grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
#grid.fit(dataset.data, dataset.target)
#print(grid)
#
# prepare a uniform distribution to sample for the alpha parameter
from lightfm.evaluation import precision_at_k
def test_sklearn_cv():

    model = LightFM(loss='warp', random_state=42)

    # Set distributions for hyperparameters
    randint = stats.randint(low=1, high=65)
    randint.random_state = 42
    gamma = stats.gamma(a=1.2, loc=0, scale=0.13)
    gamma.random_state = 42
    distr = {'no_components': randint, 'learning_rate': gamma}

    # Custom score function
    def scorer(est, x, y=None):
        return precision_at_k(est, x).mean()

    # Custom CV which sets train_index = test_index
    class CV(KFold):
        def __iter__(self):
            ind = np.arange(self.n)
            for test_index in self._iter_test_masks():
                train_index = np.logical_not(test_index)
                train_index = ind[train_index]
                yield train_index, train_index

    cv = CV(n=train.shape[0], random_state=42)
    search = RandomizedSearchCV(estimator=model, param_distributions=distr,
                                n_iter=10, scoring=scorer, random_state=42,
                                cv=cv)
    search.fit(train)
    assert search.best_params_['no_components'] == 52

test_sklearn_cv()
