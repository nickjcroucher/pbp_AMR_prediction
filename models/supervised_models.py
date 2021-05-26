from typing import Tuple, Union

from nptyping import NDArray
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso


def _fit_rf(
    train: Tuple[Union[csr_matrix, NDArray], NDArray], **kwargs
) -> RandomForestRegressor:
    kwargs = {k: round(v) for k, v in kwargs.items()}
    reg = RandomForestRegressor(**kwargs, n_jobs=-1)
    reg.fit(train[0], train[1])
    return reg


def _fit_en(
    train: Tuple[Union[csr_matrix, NDArray], NDArray],
    max_iter: int,
    l1_ratio: float = 0.5,
    alpha: float = 1.0,
) -> ElasticNet:
    reg = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=0,
        max_iter=max_iter,
    )
    reg.fit(train[0], train[1])
    return reg


def _fit_lasso(
    train: Tuple[Union[csr_matrix, NDArray], NDArray],
    max_iter: int,
    alpha: float = 1.0,
) -> Lasso:
    reg = Lasso(alpha=alpha, random_state=0, max_iter=max_iter)
    reg.fit(train[0], train[1])
    return reg
