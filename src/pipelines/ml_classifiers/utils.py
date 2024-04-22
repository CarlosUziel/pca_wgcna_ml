import json
import logging
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import shap
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from components.functional_analysis.orgdb import OrgDB
from data.ml import (
    get_gene_set_expression_data,
    process_gene_count_data,
    process_gene_sets_data,
    process_probes_meth_data,
)
from r_wrappers.complex_heatmaps import complex_heatmap, heatmap_annotation
from r_wrappers.msigdb import get_msigb_gene_sets


def process_data(
    data_type: str,
    features_type: str,
    data: Union[Path, pd.DataFrame],
    annot_df: pd.DataFrame,
    contrast_factor: str,
    org_db: OrgDB,
    custom_features_file: Path = None,
    exclude_features: Optional[Iterable[str]] = None,
    msigdb_cat: Optional[str] = None,
) -> Tuple[
    pd.DataFrame,
    Iterable[Union[str, int]],
    Iterable[Union[str, int]],
    pd.DataFrame,
    LabelEncoder,
]:
    """
    Process input data before training machine learning classifiers.

    Args:
        data_type: Nature of the input data.
        feature_type: Nature of the features to be used when modelling.
        data: A .csv file or dataframe containing data of shape [n_features, n_samples]
        annot_df: A pandas Dataframe containing samples annotations.
        contrast_factor: Column name containing the classes used for classification.
        org_db: Organism annotation database.
        custom_features_file: A .csv file where the first column is a list of relevant
            features to include.
        exclude_features: Features to remove from data.
        random_seed: Seed to initialize random state.
        msigdb_cat: MSigDB category, optional.
    """
    assert (
        len(set(annot_df[contrast_factor])) == 2
    ), "Only two clases are supported (binary classification)"

    ####################################################################################
    # 1. Data
    # 1. Get pre-processed data, class labels and overlapping features
    if data_type == "gene_expr":
        if features_type == "genes":
            (
                data_df,
                class_labels,
                overlapping_features,
                data_df_ranges,
                label_encoder,
            ) = process_gene_count_data(
                counts_file=data,
                annot_df=annot_df,
                contrast_factor=contrast_factor,
                org_db=org_db,
                custom_genes_file=custom_features_file,
                exclude_genes=exclude_features,
            )
        elif features_type == "gene_sets":
            (
                data_df,
                class_labels,
                overlapping_features,
                data_df_ranges,
                label_encoder,
            ) = get_gene_set_expression_data(
                counts=data,
                annot_df=annot_df,
                contrast_factor=contrast_factor,
                org_db=org_db,
                msigdb_cat=msigdb_cat,
                custom_genes_file=custom_features_file,
                exclude_genes=exclude_features,
            )
        else:
            raise NotImplementedError(
                f'Features type "{features_type}" is not supported for data type'
                f' "{data_type}".'
            )
    elif data_type == "probe_meth":
        if features_type == "probes":
            (
                data_df,
                class_labels,
                overlapping_features,
                data_df_ranges,
                label_encoder,
            ) = process_probes_meth_data(
                meth_values_file=data,
                annot_df=annot_df,
                contrast_factor=contrast_factor,
                org_db=org_db,
                custom_meth_probes_file=custom_features_file,
                exclude_genes=exclude_features,
            )
        else:
            raise NotImplementedError(
                f'Features type "{features_type}" is not supported for data type'
                f' "{data_type}".'
            )
    elif data_type == "gene_set_enrich":
        if features_type == "gene_sets":
            data = pd.read_csv(data, index_col=0) if isinstance(data, Path) else data
            (
                data_df,
                class_labels,
                overlapping_features,
                data_df_ranges,
                label_encoder,
            ) = process_gene_sets_data(
                data=data,
                annot_df=annot_df,
                contrast_factor=contrast_factor,
                custom_gene_sets_file=custom_features_file,
                exclude_gene_sets=exclude_features,
            )
        else:
            raise NotImplementedError(
                f'Features type "{features_type}" is not supported for data type'
                f' "{data_type}".'
            )
    else:
        raise NotImplementedError(f'Data type "{data_type}" is not supported.')

    return data_df, class_labels, overlapping_features, data_df_ranges, label_encoder


def get_best_cv_indx(cv_results: Dict) -> int:
    """
    Get index of best cross-validation results.

    Args:
        cv_results: Dictionary with CV results.

    Returns:
        Index of best result.
    """
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.sort_values(
        [
            "mean_test_balanced_accuracy",
            "mean_test_average_precision",
            "mean_test_f1",
        ],
        ascending=False,
        inplace=True,
    )
    return cv_results_df.index.to_list()[0]


def hparams_tuning(
    data_type: str,
    features_type: str,
    data: Union[Path, pd.DataFrame],
    annot_df: pd.DataFrame,
    contrast_factor: str,
    org_db: OrgDB,
    classifier_name: str,
    hparams_grid_file: Path,
    contrasts_levels_colors: Dict[str, str],
    results_path: Path,
    custom_features_file: Path,
    exclude_features: Optional[Iterable[str]] = None,
    msigdb_cat: Optional[str] = None,
    random_seed: int = 8080,
) -> None:
    """
    Hyper-parameter tuning through GridSearch of machine learning classifiers

    Args:
        classifier_name: Name of the classifier to use.
        data: A .csv file or dataframe containing data of shape [n_features, n_samples]
        annot_df: A pandas Dataframe containing samples annotations.
        contrast_factor: Column name containing the classes used for classification.
        hparams_grid_file: A .json file with a dictionary of lists, where each key is a
            hyper-paramter name and each value is a list of values to tune.
        org_db: Organism annotation database.
        contrasts_levels_colors: A mapping of contrasts levels to colors.
        results_path: Root directory to store all results.
        custom_features_file: A .csv file where the first column is a list of relevant
            features to include.
        exclude_features: Features to remove from data.
        msigdb_cat: MSigDB category, optional.
        random_seed: Seed to initialize random state.
    """
    # 0. Set seeds to ensure reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    ####################################################################################
    # 1. Data
    # 1.1. Get pre-processed data, class labels and overlapping features
    data_df, class_labels, overlapping_features, data_df_ranges, _ = process_data(
        data_type=data_type,
        features_type=features_type,
        data=data,
        annot_df=annot_df,
        contrast_factor=contrast_factor,
        org_db=org_db,
        custom_features_file=custom_features_file,
        exclude_features=exclude_features,
        msigdb_cat=msigdb_cat,
    )

    if overlapping_features.empty:
        logging.warn("There aren't any overlapping features, model training cancelled.")
        return

    ####################################################################################
    # 2. Model training
    # 2.1. Define model
    if classifier_name == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=random_seed)
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(random_state=random_seed, n_jobs=1)
    elif classifier_name == "light_gbm":
        classifier = LGBMClassifier(random_state=random_seed, n_jobs=1)
    elif classifier_name == "xgboost":
        classifier = XGBClassifier(
            eval_metric="logloss", random_state=random_seed, n_jobs=1
        )
    elif classifier_name == "nu_svc":
        classifier = NuSVC(random_state=random_seed)
    elif classifier_name == "mlp":
        classifier = MLPClassifier(random_state=random_seed)
    else:
        raise ValueError("classifier_name not valid")

    # 2.2. Logs dir
    results_path.mkdir(exist_ok=True, parents=True)

    # 2.3. Hyperparameter grid for tuning
    with hparams_grid_file.open("r") as fp:
        param_grid = json.load(fp)

    # 2.4. Hyperparameter tuning with grid-search
    scoring = ["average_precision", "precision", "recall", "f1", "balanced_accuracy"]

    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        n_jobs=-1,
        cv=5,
        verbose=0,
        scoring=scoring,
        refit=get_best_cv_indx,
    )

    # 2.4.1. Run grid search
    # `np.ascontiguousarray` improves memory usage
    grid_search.fit(
        np.ascontiguousarray(data_df), np.ravel(np.ascontiguousarray(class_labels))
    )

    ####################################################################################
    # 3. Save model and data
    model_file = results_path.joinpath(f"{classifier_name}_classifier.pkl")
    with model_file.open("wb") as fh:
        pickle.dump(grid_search, fh)

    # 3.1. Save data and annotation dataframes
    data_df.to_csv(results_path.joinpath("data.csv"))
    annot_df.to_csv(results_path.joinpath("annot_data.csv"))

    # 3.2. Save genes without overlap and plot heatmap
    non_overlapping_features = overlapping_features[~overlapping_features]
    if not non_overlapping_features.empty:
        if features_type == "genes" or features_type == "probes":
            custom_features_df = pd.read_csv(
                custom_features_file, index_col=0, dtype={"ENTREZID": str}
            )

            if features_type == "genes":
                custom_features_df.dropna(subset=["ENTREZID"], inplace=True)

                non_overlapping_features_df = custom_features_df[
                    custom_features_df["ENTREZID"].isin(
                        non_overlapping_features.index.astype(str)
                    )
                ]

                non_overlapping_features_df.to_csv(
                    results_path.joinpath("non_overlapping_features.csv")
                )

                counts_matrix = pd.read_csv(data, index_col=0).loc[
                    non_overlapping_features_df.index,
                    data_df.index,
                ]

            else:
                custom_features_df.loc[non_overlapping_features.index, :].to_csv(
                    results_path.joinpath("non_overlapping_features.csv")
                )

                counts_matrix = pd.read_csv(data, index_col=0).loc[
                    non_overlapping_features.index,
                    data_df.index,
                ]

            # substract row means for better visualization
            counts_matrix = counts_matrix.sub(counts_matrix.mean(axis=1), axis=0)

            ha_column = heatmap_annotation(
                df=annot_df[[contrast_factor]],
                col={contrast_factor: contrasts_levels_colors},
            )

            complex_heatmap(
                counts_matrix,
                save_path=results_path.joinpath(
                    "non_overlapping_features_clustering.pdf"
                ),
                width=10,
                height=10,
                name=f"Non-overlapping features ({features_type})",
                column_title=f"Features (N={len(counts_matrix)})",
                top_annotation=ha_column,
                show_row_names=False,
                show_column_names=False,
                cluster_columns=False,
                heatmap_legend_param=ro.r(
                    'list(title_position = "topcenter", color_bar = "continuous",'
                    ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
                ),
            )

        else:
            non_overlapping_features.to_csv(
                results_path.joinpath("non_overlapping_features.csv")
            )

    # 3.3. Save gene counts ranges
    data_df_ranges.to_csv(results_path.joinpath("data_ranges.csv"))

    # 3.4. Save parameters of best estimator
    with results_path.joinpath("best_hparams.json").open("w") as fh:
        json.dump(
            grid_search.best_estimator_.get_params(deep=True),
            fh,
            indent=4,
            sort_keys=True,
        )

    ####################################################################################
    # 4. Evaluation
    # 4.1. cross-validation results
    pd.DataFrame(grid_search.cv_results_).sort_values(
        [
            "mean_test_balanced_accuracy",
            "mean_test_average_precision",
            "mean_test_f1",
        ],
        ascending=False,
    ).to_csv(results_path.joinpath("cv_results.csv"))


def bootstrap_relevant_features(
    counts_df: pd.DataFrame,
    class_labels: Iterable[int],
    model: Any,
    random_seeds: Union[int, Iterable[int]] = 100,
) -> pd.DataFrame:
    """
    Trains a model multiple times, with different random seeds, and saves the most
    relevant features by storing its importance score in a dataframe, one row per
    random seed. Those features that were not "chosen" in a certain training run will
    have importance scores equal to 0.

    Args:
        counts_df: Data in the shape of [n_samples, n_genes]
        class_labels: List of class labels.
        model: Base model to train.
        random_seeds: Number of random seeds (bootstrap iterations) to run.

    Returns:
        A pd.DataFrame with the feature importances per random run.
    """
    # 0. Setup
    test_scores = defaultdict(dict)
    shap_values_mean = None

    # 1. Choose random_seeds
    random_seeds = (
        random_seeds
        if type(random_seeds) == List
        else random.sample(range(random_seeds * random_seeds), random_seeds)
    )

    # 2. Bootstrap training
    for i, random_seed in enumerate(random_seeds):
        # 2.1. Set seeds to ensure reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 2.2. Split data into training and testing data sets
        train_data, test_data, train_labels, test_labels = train_test_split(
            counts_df,
            class_labels,
            train_size=0.8,
            random_state=random_seed,
            stratify=class_labels,
        )

        # 2.3. Model training
        # `np.ascontiguousarray` improves memory usage
        model = deepcopy(model)
        model.random_state = random_seed
        model.fit(
            np.ascontiguousarray(train_data),
            np.ascontiguousarray(train_labels),
        )

        # 2.4. Score on test set
        test_pred = model.predict(np.ascontiguousarray(test_data))
        test_scores[random_seed]["average_precision"] = average_precision_score(
            np.ascontiguousarray(test_labels),
            test_pred,
        )
        test_scores[random_seed]["F1S"] = f1_score(
            np.ascontiguousarray(test_labels),
            test_pred,
        )
        test_scores[random_seed]["balanced_accuracy"] = balanced_accuracy_score(
            np.ascontiguousarray(test_labels),
            test_pred,
        )

        # 2.5. Get SHAP values
        try:
            explainer = shap.Explainer(model, algorithm="auto", n_jobs=1)
            shap_values = explainer(
                X=np.ascontiguousarray(counts_df), y=np.ascontiguousarray(class_labels)
            ).values
        except Exception:
            explainer = shap.KernelExplainer(
                model.predict, np.ascontiguousarray(counts_df)
            )
            shap_values = explainer.shap_values(np.ascontiguousarray(counts_df))

        shap_values = (
            shap_values[:, :, 1] if len(shap_values.shape) > 2 else shap_values
        )

        # running average
        # new_average = (old_average * (n-1) + new_value) / n
        shap_values_mean = (
            (shap_values_mean * i + shap_values) / (i + 1)
            if shap_values_mean is not None
            else shap_values
        )

    return test_scores, shap_values_mean


def bootstrap_training(
    data_type: str,
    features_type: str,
    data: Union[Path, pd.DataFrame],
    annot_df: pd.DataFrame,
    contrast_factor: str,
    org_db: OrgDB,
    classifier_name: str,
    hparams_file: Path,
    results_path: Path,
    custom_features_file: Path,
    exclude_features: Optional[Iterable[str]] = None,
    msigdb_cat: Optional[str] = None,
    bootstrap_iterations: int = 1000,
    random_seed: int = 8080,
) -> None:
    """
    Bootstrap training of binary classifiers.

    Args:
        classifier_name: Name of the classifier to use.
        data: A .csv file or dataframe containing data of shape [n_features, n_samples]
        annot_df: A pandas Dataframe containing samples annotations.
        contrast_factor: Column name containing the classes used for classification.
        hparams_file: .json file containing the hyper-parameters to train the model.
        org_db: Organism annotation database.
        results_path: Root directory to store all results.
        custom_features_file: A .csv file where the first column is a list of relevant
            features to include.
        exclude_features: Features to remove from data.
        msigdb_cat: MSigDB category, optional.
        bootstrap_iterations: Number of random seeds to generate.
        random_seed: Seed to initialize random state.
    """
    # 0. Set seeds to ensure reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    ####################################################################################
    # 1. Data
    # 1.1. Get pre-processed data, class labels and overlapping features
    data_df, class_labels, overlapping_features, _, label_encoder = process_data(
        data_type=data_type,
        features_type=features_type,
        data=data,
        annot_df=annot_df,
        contrast_factor=contrast_factor,
        org_db=org_db,
        custom_features_file=custom_features_file,
        exclude_features=exclude_features,
        msigdb_cat=msigdb_cat,
    )

    if overlapping_features.empty:
        logging.warn("There aren't any overlapping features, model training cancelled.")
        return

    ####################################################################################
    # 2. Model training
    # Define model - Hyper-parameters extracted from grid-search tuning
    with hparams_file.open("r") as fh:
        if classifier_name == "decision_tree":
            classifier = DecisionTreeClassifier(**json.load(fh))
        elif classifier_name == "random_forest":
            classifier = RandomForestClassifier(**json.load(fh))
        elif classifier_name == "light_gbm":
            classifier = LGBMClassifier(**json.load(fh))
        elif classifier_name == "xgboost":
            classifier = XGBClassifier(**json.load(fh))
        elif classifier_name == "nu_svc":
            classifier = NuSVC(**json.load(fh))
        elif classifier_name == "mlp":
            classifier = MLPClassifier(**json.load(fh))
        else:
            raise ValueError("classifier_name not valid")

    test_scores, shap_values = bootstrap_relevant_features(
        counts_df=data_df,
        class_labels=class_labels,
        model=classifier,
        random_seeds=bootstrap_iterations,
    )

    ####################################################################################
    # 3. Aggregate and annotate bootstrap results
    results_path.mkdir(exist_ok=True, parents=True)

    # 3.1. Feature annotations
    custom_features_df = pd.read_csv(
        custom_features_file, index_col=0, dtype={"ENTREZID": str}
    )
    feature_annotations = None
    if data_type == "gene_expr":
        if features_type == "genes":
            feature_annotations = (
                custom_features_df.dropna(subset=["ENTREZID"])
                .drop_duplicates(subset=["ENTREZID"], keep=False)
                .set_index("ENTREZID")
                .loc[data_df.columns.astype(str)]
            )
            feature_names = feature_annotations["SYMBOL"]
        elif features_type == "gene_sets":
            feature_annotations = pd.concat(
                (
                    pd.Series(
                        get_msigb_gene_sets(
                            species=org_db.species,
                            category=msigdb_cat,
                            gene_id_col="entrez_gene",
                        ),
                        name="degs_entrez",
                    ).apply(
                        lambda x: set(
                            custom_features_df["ENTREZID"]
                            .dropna()
                            .drop_duplicates(keep=False)
                        ).intersection(x)
                    ),
                    pd.Series(
                        get_msigb_gene_sets(
                            species=org_db.species,
                            category=msigdb_cat,
                            gene_id_col="gene_symbol",
                        ),
                        name="degs_symbol",
                    ).apply(
                        lambda x: set(
                            custom_features_df["SYMBOL"]
                            .dropna()
                            .drop_duplicates(keep=False)
                        ).intersection(x)
                    ),
                ),
                axis=1,
            ).applymap(lambda x: "/".join(map(str, x)))
            feature_names = data_df.columns
    elif data_type == "probe_meth":
        if features_type == "probes":
            feature_annotations = custom_features_df.loc[
                data_df.columns,
                [
                    "annot.seqnames",
                    "annot.start",
                    "annot.end",
                    "annot.width",
                    "annot.strand",
                    "annot.id",
                    "annot.tx_id",
                    "annot.gene_id",
                    "annot.symbol",
                    "annot.type",
                ],
            ].dropna(subset="annot.gene_id")
            feature_names = feature_annotations["annot.symbol"]

    elif data_type == "gene_set_enrich":
        if features_type == "gene_sets":
            feature_annotations = custom_features_df.loc[
                data_df.columns,
                ["entrez_gene", "gene_symbol"],
            ]
            feature_names = data_df.columns

    if feature_annotations is None:
        raise NotImplementedError(
            f"No available annotation for features types {features_type} of data type"
            f" {data_type}"
        )

    # 3.2 Process shap values
    # 3.2.1. Plots
    for max_display in (10, 30, 50):
        # Beeswarm plot
        shap.summary_plot(
            shap_values,
            data_df,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.suptitle("Top features contributing to binary model output", fontsize=20)
        plt.title(
            f"{label_encoder.classes_[0]} \u27F7 {label_encoder.classes_[1]}",
            fontsize=14,
            pad=20,
        )
        plt.gcf().axes[-1].set_aspect("auto")
        plt.tight_layout()
        plt.gcf().axes[-1].set_box_aspect(100)

        plt.savefig(
            results_path.joinpath(
                Path(
                    f"bootstrap_{bootstrap_iterations}_shap_values_beeswarm_plot_"
                    f"top_{max_display}.pdf"
                )
            )
        )
        plt.close()

    # 3.2.2. Annotate and save
    pd.DataFrame(
        shap_values, index=data_df.index, columns=data_df.columns.astype(str)
    ).transpose().to_csv(
        results_path.joinpath(Path(f"bootstrap_{bootstrap_iterations}_shap_matrix.csv"))
    )
    shap_values_df = pd.concat(
        [
            pd.DataFrame(
                {"shap_value": np.abs(shap_values).mean(axis=0)},
                index=data_df.columns.astype(str),
            ),
            feature_annotations,
        ],
        axis=1,
    ).sort_values("shap_value", key=abs, ascending=False)

    shap_values_df.to_csv(
        results_path.joinpath(Path(f"bootstrap_{bootstrap_iterations}_shap_values.csv"))
    )

    # 3.2.3. Save top genes at different shap thresholds
    shap_values_filtered = {}
    for shap_thr in (1e-03, 1e-04, 1e-05):
        shap_values_filtered[shap_thr] = shap_values_df[
            shap_values_df["shap_value"] > shap_thr
        ]

    for shap_thr, degs_scores_df in shap_values_filtered.items():
        shap_thr_str = str(shap_thr).replace(".", "_")
        degs_scores_df.to_csv(
            results_path.joinpath(
                f"bootstrap_{bootstrap_iterations}_shap_values_{shap_thr_str}.csv"
            )
        )

    # 3.3. Save test scores
    pd.DataFrame(test_scores).transpose().sort_values(
        ["balanced_accuracy", "average_precision", "F1S"], ascending=False
    ).to_csv(
        results_path.joinpath(Path(f"bootstrap_{bootstrap_iterations}_test_scores.csv"))
    )
