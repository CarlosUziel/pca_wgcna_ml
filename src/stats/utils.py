from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from data.utils import powerset
from stats.ml import evaluate_logistic_regression


def evaluate_biomarkers(
    annot_df: pd.DataFrame,
    biomarkers: Iterable[Union[str, int]],
    cond_col: str,
    save_path: Path,
):
    """
    Evaluate precision and recall of the powerset of biomarkers in a condition
    classification task.

    Args:
        annot_df: An annotation dataframe containing columns of biomarker gene
            expression.
        biomarkers: List of biomarkers to evaluate.
        cond_col: Column name defining sample category (e.g., sample type,
            patient condition, etc.)
        save_path: CSV file name to store evaluation results.
    """
    assert save_path.suffix == ".csv", "save_path must point to a .csv file"

    # 1. Cross-validate prediction metrics of the biomarkers powerset
    biomarkers_eval_results = {}
    for genes in list(powerset(biomarkers))[1:]:
        cv_scores_df = evaluate_logistic_regression(annot_df, biomarkers, cond_col)
        biomarkers_eval_results["_".join(genes)] = {
            metric: f"{mean:.2f} \u00b1 {std:.2f}"
            for metric, mean, std in zip(
                cv_scores_df.columns,
                cv_scores_df.mean().values,
                cv_scores_df.std().values,
            )
        }

    # 2. Format results and save
    pd.DataFrame(biomarkers_eval_results).transpose().to_csv(save_path)


def iou(list_1: Iterable, list_2: Iterable):
    """Compute the Jaccard index or Intersection over Union for two sets"""
    intersection = set(list_1).intersection(set(list_2))
    union = set(list_1).union(set(list_2))
    return len(intersection) / len(union)


# overlap coefficient (https://en.wikipedia.org/wiki/Overlap_coefficient)
# The Overlap Coefficient is recommended when relations are expected to occur between
# large-size and small-size gene-sets, as in the case of the Gene Ontology.
# The Jaccard Coefficient is recommended in the opposite case.
def overlap(list_1: Iterable, list_2: Iterable):
    """Compute the Overlap Coefficient for two sets

    The Overlap Coefficient is recommended when relations are expected to occur between
        large-size and small-size gene-sets, as in the case of the Gene Ontology.
        The Jaccard Coefficient is recommended in the opposite case.
    """
    intersection = set(list_1).intersection(set(list_2))
    return len(intersection) / min(len(set(list_1)), len(set(list_2)))
