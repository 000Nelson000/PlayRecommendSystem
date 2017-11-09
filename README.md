# Recommended System Play-Ground. 
This is the place I try to figure out how to implement recommendation algos

## data sets
- movielens
- funds
- sketchFab

## Algos

-  nearest neighbor (User-Based CF, Item-Based CF)
	- [ ] issue: very inefficient to compute pairwise similarity(distance)	

- explicit matrix factorization (ALS, SGD ...)

- implicit matrix factorization 

- web-based funds recommendation, [demo code](https://github.com/ihongChen/PlayRecommendSystem/tree/master/funds_web)

- [v] LightFM 
	
- Approximate Nearest Neighbors
	-  issue: can't install properyly under windows env (solved)
		- annoy -- ubuntu16.04, windows server2012 (update conda to python 3.6 version)
		- nmslib -- failed
	-  how to use it properly?
		- [example](https://github.com/ihongChen/PlayRecommendSystem/blob/master/LightFM_ANN_example.py)

## Notebooks

