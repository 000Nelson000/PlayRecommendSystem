# Recommended System Play-Ground. 
This is the place I try to figure out how to implement recommendation algos

## data sets
- movielens
- funds
- sketchFab
- lastfm

## Algos
- content based [demo code](https://github.com/ihongChen/PlayRecommendSystem/blob/master/cb_funds.ipynb)
- nearest neighbor (User-Based CF, Item-Based CF)
	- issue: very inefficient to compute pairwise similarity(distance)
	- [demo code](https://github.com/ihongChen/PlayRecommendSystem/blob/master/knn_funds_recommendation.py)
- matrix factorization (explicit/implicit ALS, SGD ...) [demo](https://github.com/ihongChen/PlayRecommendSystem/blob/master/rec_funds/rec_funds_offline.ipynb)

- Learning to Rank 
	- LightFM [coldstart](https://github.com/ihongChen/PlayRecommendSystem/blob/master/03_lightfm_coldstart.py)
### speed up	
- Approximate Nearest Neighbors
	-  issue: can't install properyly under windows env (solved)
		- annoy -- ubuntu16.04, windows server2012 (update conda to python 3.6 version)
		- nmslib -- failed
	-  how to use it?
		[lightFM+annoy](https://github.com/ihongChen/PlayRecommendSystem/blob/master/08_lightfm_annoy_funds.ipynb)

## Demo website
- web-based funds recommendation, [demo code](https://github.com/ihongChen/PlayRecommendSystem/tree/master/funds_web)


## Notebooks

- [x] [introduction to recommendation - KNN model](https://github.com/ihongChen/PlayRecommendSystem/blob/master/notebook/rec_note1_knn.ipynb)
- [ ] [matrix factorization - ALS](https://github.com/ihongChen/PlayRecommendSystem/blob/master/notebook/rec_note2_als.ipynb)
- [ ] learning to rank 
- [ ] content based 