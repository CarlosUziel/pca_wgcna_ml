import multiprocessing
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder


def evaluate_logistic_regression(
    data: pd.DataFrame,
    features_cols: Iterable[str],
    targets_col: str,
    random_seed: int = 8080,
    **kwargs,
) -> pd.DataFrame:
    """
    Evaluate a logistic regression model through cross-validation.

    Args:
        data: Dataframe containing dependent and independent variables.
        features_cols: Names of the columns to be used as features/independent
            variables.
        targets_col. Name of the targets column.
        random_seed: Seed to set random state.

    Returns:
        A pandas dataframe object containing CV results.
    """

    # 1. Setup inputs
    X = np.ascontiguousarray(data[features_cols])
    y = LabelEncoder().fit_transform(data[targets_col])

    # 3. Cross validation metrics estimation
    # Use LogisticRegressionCV estimator to get best hyper-parameters every time
    cv_scores = cross_validate(
        LogisticRegressionCV(
            random_state=random_seed,
            scoring="f1_weighted",
            class_weight="balanced",
            n_jobs=multiprocessing.cpu_count() - 2,
            **kwargs,
        ),
        X,
        y,
        scoring=[
            "f1_weighted",
            "precision_weighted",
            "recall_weighted",
            "roc_auc_ovo_weighted",
        ],
        cv=10,
    )

    return pd.DataFrame(cv_scores)
