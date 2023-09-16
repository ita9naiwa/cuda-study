import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from als import cyALS
import time
import numpy as np
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import train_test_split, ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares as ALS
_, ratings = get_movielens('100k')
ratings = ratings.T.tocsr()
tr, te = train_test_split(ratings, 0.8)

dim = 32
reg = 100.0


def model_eval(X, Y, tr, te, K=10):
    """
        this exploits implicit's evaluation features
    """
    model = ALS(use_gpu=False)
    model.user_factors = X
    model.item_factors = Y
    return ranking_metrics_at_k(model, tr, te, K=10)

# implicit = ALS(factors=4096, iterations=5, use_gpu=True, use_cg=False, num_threads=0)
# prev = time.time()
# implicit.fit(tr)
# print("runtime", time.time() - prev)
# print(ranking_metrics_at_k(implicit, tr, te))

# implicit = ALS(factors=dim, iterations=5, use_gpu=False, use_cg=False, num_threads=0)
# prev = time.time()
# implicit.fit(tr)
# print("runtime", time.time() - prev)
# implicit = ALS(factors=dim, iterations=5, use_gpu=False, use_cg=True, num_threads=0)
# prev = time.time()
# implicit.fit(tr)
# print("runtime", time.time() - prev)

# prev = time.time()
# X, Y = cyALS(tr, d=dim, reg=reg, max_iter=5, num_threads=0, method='cpu_naive')
# print("runtime", time.time() - prev)
# print(model_eval(X, Y, tr, te))


# prev = time.time()
# X, Y = cyALS(tr, d=dim, reg=reg, max_iter=5, num_threads=0, method='cpu_cg')
# print("runtime", time.time() - prev)
# print(model_eval(X, Y, tr, te))


dim = 32
reg = 0
iter = 1
prev = time.time()
X, Y = cyALS(tr, d=dim, reg=reg, max_iter=iter, num_threads=0, method='cpu_cg')
print("runtime", time.time() - prev)
print(model_eval(X, Y, tr, te))



prev = time.time()
X, Y = cyALS(tr, d=dim, reg=reg, max_iter=iter, num_threads=0, method='gpu_naive')
print("runtime", time.time() - prev)
print(model_eval(X, Y, tr, te))
