from als import cyALS
import numpy as np
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import train_test_split, ranking_metrics_at_k
from implicit.als import AlternatingLeastSquares as ALS
_, ratings = get_movielens('100k')
ratings = ratings.T.tocsr()
tr, te = train_test_split(ratings, 0.8)
X, Y = cyALS(tr, d=32, reg=1.0, max_iter=15)

def model_eval(X, Y, tr, te, K=10):
    """
        this exploits implicit's evaluation features
    """
    model = ALS(use_gpu=False)
    model.user_factors = X
    model.item_factors = Y
    return ranking_metrics_at_k(model, tr, te, K=10)

print(model_eval(X, Y, tr, te, K=10))