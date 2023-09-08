from als import cyALS
import numpy as np
from implicit.datasets.movielens import get_movielens

_, ratings = get_movielens('100k')
ratings = ratings.T.tocsr()
print(type(ratings))
print(ratings.indices.shape, ratings.indices)
print(ratings.indptr.shape, ratings.indptr)
print(np.ediff1d(ratings.indices))
print(ratings.shape)

cyALS(ratings, d=32, reg=1.0, max_iter=10)