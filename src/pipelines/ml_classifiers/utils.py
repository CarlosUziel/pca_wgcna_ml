import json
import logging
import pickle
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from components.functional_analysis.orgdb import OrgDB
from data.ml import process_gene_count_data

# from r_wrappers.complex_heatmaps import complex_heatmap, heatmap_annotation

warnings.filterwarnings(
    "ignore", message="No further splits with positive gain", category=UserWarning
)


def get_best_cv_indx(cv_results: Dict[str, np.ndarray]) -> int:
    """
    Get index of best cross-validation results based on multiple metrics.

    Args:
        cv_results: Dictionary with cross-validation results from GridSearchCV

    Returns:
        Index of best result after sorting by balanced accuracy, F1, precision and recall
    """
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.sort_values(
        [
            "mean_test_balanced_accuracy",
            "mean_test_f1",
            "mean_test_precision",
            "mean_test_recall",
        ],
        ascending=False,
        inplace=True,
    )
    return cv_results_df.index.to_list()[0]


def get_classifier(
    classifier_name: str, random_seed: int, n_jobs: int = 1, **kwargs
) -> Union[
    RandomForestClassifier,
    DecisionTreeClassifier,
    XGBClassifier,
    LGBMClassifier,
    NuSVC,
    MLPClassifier,
    TabPFNClassifier,
]:
    """Create and configure a tree-based classifier instance for binary classification.

    Args:
        classifier_name: Name of classifier to create. One of:
            - 'decision_tree'
            - 'random_forest'
            - 'light_gbm'
            - 'xgboost'
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs for supported classifiers
        **kwargs: Additional parameters to pass to classifier

    Returns:
        Configured tree-based classifier instance

    Raises:
        ValueError: If classifier_name is not one of the supported tree-based classifiers
    """
    # Set default parameters
    params = kwargs.copy()
    params.setdefault("random_state", random_seed)

    if classifier_name in ["random_forest", "light_gbm", "xgboost"]:
        params.setdefault("n_jobs", n_jobs)

    # Configure classifier-specific defaults
    if classifier_name == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif classifier_name == "random_forest":
        return RandomForestClassifier(**params)
    elif classifier_name == "light_gbm":
        params.setdefault("device", "cpu")
        params.setdefault("verbose", -1)
        params.setdefault("min_gain_to_split", 0)
        return LGBMClassifier(**params)
    elif classifier_name == "xgboost":
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cpu")
        return XGBClassifier(**params)
    elif classifier_name == "nu_svc":
        params.setdefault("probability", True)
        return NuSVC(**params)
    elif classifier_name == "mlp":
        return MLPClassifier(**params)
    elif classifier_name == "tabpfn":
        params.setdefault("device", "cpu")
        return TabPFNClassifier(**params)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")


def get_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate binary classification metrics.

    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)

    Returns:
        Dictionary of metric names to values
    """
    return {
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _validate_binary_classification(class_labels: np.ndarray) -> None:
    """Validate that labels represent a binary classification problem.

    Args:
        class_labels: Array of class labels, expected to contain exactly 2 unique values

    Raises:
        ValueError: If not exactly 2 unique classes are found, with details about found classes
    """
    unique_classes = np.unique(class_labels)
    if len(unique_classes) != 2:
        raise ValueError(
            "Expected binary classification (2 classes), "
            f"got {len(unique_classes)} classes: {unique_classes}"
        )


def hparams_tuning(
    data: Union[Path, pd.DataFrame],
    annot_df: pd.DataFrame,
    contrast_factor: str,
    org_db: OrgDB,
    classifier_name: str,
    hparams_grid_file: Path,
    contrasts_levels_colors: Dict[str, str],
    results_path: Path,
    custom_features: Optional[pd.DataFrame] = None,
    custom_features_gene_type: Optional[str] = "ENTREZID",
    exclude_features: Optional[Iterable[str]] = None,
    n_jobs: int = 1,
    random_seed: int = 8080,
) -> None:
    """Perform hyperparameter tuning through GridSearch for binary classification.

    Args:
        data: Gene expression matrix [n_features, n_samples] as file or DataFrame
        annot_df: Sample annotations with binary class labels
        contrast_factor: Column in annot_df containing binary class labels
        org_db: Organism annotation database
        classifier_name: Name of classifier ('decision_tree', 'random_forest', etc.)
        hparams_grid_file: JSON file containing parameter grid for GridSearchCV
        contrasts_levels_colors: Mapping of class labels to colors for plots
        results_path: Directory to save results (models, plots, metrics)
        custom_features: DataFrame with gene annotations having ENTREZID as index
        custom_features_gene_type: Type of gene IDs in custom_features index
        exclude_features: Features to exclude from analysis
        n_jobs: Number of parallel jobs for CV and supported classifiers
        random_seed: Random seed for reproducibility

    Notes:
        - Only binary classification is supported
        - Input data dimensions must match [n_features, n_samples]
        - n_jobs affects GridSearchCV and tree-based classifiers
    """
    # 0. Set seeds to ensure reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    ####################################################################################
    # 1. Data
    # 1.1. Get pre-processed data, class labels and overlapping features
    data_df, class_labels, overlapping_features, data_df_ranges = (
        process_gene_count_data(
            counts=data,
            annot_df=annot_df,
            contrast_factor=contrast_factor,
            org_db=org_db,
            custom_features=custom_features,
            custom_features_gene_type=custom_features_gene_type,
            exclude_features=exclude_features,
        )
    )
    # Save label id to string mapping for reproducibility
    label_mapping = {
        int(idx): str(label)
        for idx, label in enumerate(
            sorted(
                set(annot_df[contrast_factor]),
                key=list(annot_df[contrast_factor]).index,
            )
        )
    }
    with results_path.joinpath("label_mapping.json").open("w") as fh:
        json.dump(label_mapping, fh, indent=4)
    _validate_binary_classification(class_labels)

    if overlapping_features.empty:
        logging.warning(
            "There aren't any overlapping features, model training cancelled."
        )
        return

    ####################################################################################
    # 2. Model training
    # 2.1. Define model
    classifier = get_classifier(classifier_name, random_seed, n_jobs)

    # 2.2. Logs dir
    results_path.mkdir(exist_ok=True, parents=True)

    # 2.3. Hyperparameter grid for tuning
    with hparams_grid_file.open("r") as fp:
        param_grid = json.load(fp)

    # 2.4. Hyperparameter tuning with grid-search
    scoring = [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]

    # Suppress specific warnings during grid search
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            n_jobs=n_jobs,
            cv=5,
            verbose=0,
            scoring=scoring,
            refit=get_best_cv_indx,
        )

        # 2.4.1. Run grid search with warning suppression
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
    # non_overlapping_features = overlapping_features[~overlapping_features]
    # if not non_overlapping_features.empty and custom_features is not None:
    #     non_overlapping_features_df = custom_features[
    #         custom_features.index.isin(non_overlapping_features.index.astype(str))
    #     ]

    #     if isinstance(data, Path):
    #         counts_matrix_full = pd.read_csv(data, index_col=0)
    #     else:
    #         counts_matrix_full = data.copy()

    #     # Ensure indices are strings for consistent matching
    #     counts_matrix_full.index = counts_matrix_full.index.astype(str)
    #     counts_matrix_full.columns = counts_matrix_full.columns.astype(str)
    #     non_overlapping_features_df.index = non_overlapping_features_df.index.astype(
    #         str
    #     )
    #     data_df.index = data_df.index.astype(str)

    #     row_indexer = non_overlapping_features_df.index.intersection(
    #         counts_matrix_full.index
    #     ).tolist()
    #     col_indexer = data_df.index.intersection(counts_matrix_full.columns).tolist()

    #     counts_matrix = counts_matrix_full.loc[row_indexer, col_indexer]

    #     substract row means for better visualization
    #     if not counts_matrix.empty:
    #         counts_matrix = counts_matrix.sub(counts_matrix.mean(axis=1), axis=0)

    #         ha_column = heatmap_annotation(
    #             df=annot_df[[contrast_factor]],
    #             col={contrast_factor: contrasts_levels_colors},
    #         )

    #         complex_heatmap(
    #             counts_matrix,
    #             save_path=results_path.joinpath(
    #                 "non_overlapping_features_clustering.pdf"
    #             ),
    #             width=10,
    #             height=10,
    #             name="Non-overlapping genes",
    #             column_title=f"Features (N={len(counts_matrix)})",
    #             top_annotation=ha_column,
    #             show_row_names=False,
    #             show_column_names=False,
    #             cluster_columns=False,
    #             heatmap_legend_param=ro.r(
    #                 'list(title_position = "topcenter", '
    #                 'color_bar = "continuous", '
    #                 'legend_height = unit(5, "cm"), '
    #                 'legend_direction = "horizontal")'
    #             ),
    #         )

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
            "mean_test_f1",
            "mean_test_precision",
            "mean_test_recall",
        ],
        ascending=False,
    ).to_csv(results_path.joinpath("cv_results.csv"))


@dataclass
class ShapResults:
    """Container for SHAP values from tree-based binary classifiers.

    Attributes:
        values: Main SHAP values with shape [n_samples, n_features],
               representing direct contribution of each feature to model output
        interaction_values: SHAP interaction values with shape
                          [n_samples, n_features, n_features], representing
                          how features interact with each other
    """

    values: np.ndarray
    interaction_values: np.ndarray


def bootstrap_relevant_features(
    counts_df: pd.DataFrame,
    class_labels: np.ndarray,
    classifier_name: str,
    model_params: Dict,
    random_seeds: Union[int, Iterable[int]] = 100,
    n_jobs: int = 1,
) -> Tuple[Dict[int, Dict[str, float]], ShapResults]:
    """Train a tree-based model multiple times to compute SHAP values and interactions.

    Args:
        counts_df: Input feature matrix [n_samples, n_features]
        class_labels: Binary class labels [n_samples], must contain exactly 2 unique values
        classifier_name: Name of the classifier to use.
        model_params: Parameters for the classifier.
        random_seeds: Number of iterations or list of random seeds to use
        n_jobs: Number of parallel jobs for supported classifiers.

    Returns:
        Tuple containing:
        - Dict mapping random seeds to metrics (precision, recall, f1, balanced_accuracy)
        - ShapResults containing averaged SHAP values across all iterations:
            - values: Feature contributions [n_samples, n_features]
            - interaction_values: Feature interactions [n_samples, n_features, n_features]

    Raises:
        ValueError: If class_labels contains != 2 classes or model lacks predict_proba

    Notes:
        - Only tree-based models are supported for SHAP TreeExplainer compatibility
        - Uses class 1 (positive class) SHAP values for binary classification
        - Uses running average to compute final values using iteration index
        - Any iteration failure will stop the entire process
    """
    # 0. Setup and validation
    _validate_binary_classification(class_labels)

    test_scores = defaultdict(dict)

    # 1. Choose random_seeds
    seeds: Iterable[int]
    if isinstance(random_seeds, int):
        seeds = random.sample(range(random_seeds * 10), random_seeds)
    else:
        seeds = random_seeds

    # Initialize outputs with None
    shap_values_mean = None
    shap_interactions_mean = None

    # 2. Bootstrap training
    for iteration, random_seed in enumerate(seeds):
        # 2.1. Set seeds
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
        model_params = dict(model_params)
        model_params.pop("n_jobs", None)
        current_model = get_classifier(
            classifier_name, random_seed, n_jobs, **model_params
        )
        if not hasattr(current_model, "predict_proba"):
            raise ValueError(
                f"Model {current_model.__class__.__name__} must support probability "
                "predictions for SHAP values"
            )

        current_model.fit(
            np.ascontiguousarray(train_data),
            np.ascontiguousarray(train_labels),
        )

        # 2.4. Score on test set and verify predictions
        test_pred = current_model.predict(np.ascontiguousarray(test_data))
        test_scores[random_seed].update(get_model_metrics(test_labels, test_pred))

        # Use training data as background for SHAP
        background = (
            train_data
            if len(train_data) <= 1000
            else train_data.sample(1000, random_state=random_seed)
        )

        # Get SHAP values and verify shapes immediately
        # Use probability output for main SHAP values
        explainer = shap.TreeExplainer(
            current_model, data=background, model_output="probability"
        )
        shap_values = explainer.shap_values(counts_df)

        # Use raw output for SHAP interaction values (required by SHAP)
        explainer_inter = shap.TreeExplainer(current_model, model_output="raw")
        shap_interaction_values = explainer_inter.shap_interaction_values(counts_df)

        # Handle class dimension in SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Select SHAP values for the positive class
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]  # Select positive class

        if (
            isinstance(shap_interaction_values, list)
            and len(shap_interaction_values) == 2
        ):
            shap_interaction_values = shap_interaction_values[
                1
            ]  # Select SHAP values for the positive class
        elif isinstance(shap_interaction_values, np.ndarray) and shap_interaction_values.ndim == 4 and shap_interaction_values.shape[3] == 2:
            shap_interaction_values = shap_interaction_values[:, :, :, 1]  # Select positive class

        # Verify shapes after handling class dimension
        assert isinstance(shap_values, np.ndarray), (
            "SHAP values should be a numpy array"
        )
        assert shap_values.ndim == 2, (
            f"Unexpected SHAP values shape: {shap_values.shape}"
        )
        assert isinstance(shap_interaction_values, np.ndarray), (
            "SHAP interaction values should be a numpy array"
        )
        assert shap_interaction_values.ndim == 3, (
            f"Unexpected SHAP interaction values shape: {shap_interaction_values.shape}"
        )

        # Update running averages
        if iteration == 0:
            shap_values_mean = shap_values
            shap_interactions_mean = shap_interaction_values
        else:
            assert (
                shap_values_mean is not None and shap_interactions_mean is not None
            ), "Mean SHAP values should have been initialized"
            shap_values_mean = (shap_values_mean * iteration + shap_values) / (
                iteration + 1
            )
            shap_interactions_mean = (
                shap_interactions_mean * iteration + shap_interaction_values
            ) / (iteration + 1)

    assert shap_values_mean is not None and shap_interactions_mean is not None, (
        "Mean SHAP values were not computed"
    )
    return test_scores, ShapResults(shap_values_mean, shap_interactions_mean)


def _sanitize_filename(text: str) -> str:
    """Convert text to safe filename by replacing non-alphanumeric chars with underscore.

    Args:
        text: Input text to sanitize

    Returns:
        String with only alphanumeric characters and underscores
    """
    return "".join(c if c.isalnum() else "_" for c in text)


def create_shap_plots(
    shap_values: np.ndarray,
    data_df: pd.DataFrame,
    feature_names: pd.Series,
    save_path: Path,
    prefix: str,
    max_display: int = 30,
) -> Dict[str, bool]:
    """Create standard SHAP visualization plots.

    Args:
        shap_values: SHAP values [n_samples, n_features]
        data_df: Original feature matrix [n_samples, n_features]
        feature_names: Names of features [n_features]
        save_path: Directory to save plots
        prefix: Prefix for plot filenames
        max_display: Maximum number of features to display in summary plots

    Returns:
        Dictionary mapping plot types to success status
    """
    plot_results = {}

    # Create subdirectory for dependence plots
    dependence_dir = save_path.joinpath(f"{prefix}_dependence")
    dependence_dir.mkdir(exist_ok=True, parents=True)

    # 1. Summary plots with same logic as before
    try:
        # Violin plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            data_df,
            feature_names=feature_names.tolist(),
            max_display=max_display,
            plot_type="violin",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_summary_violin.pdf"))
        plt.close()

        # Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            data_df,
            feature_names=feature_names.tolist(),
            max_display=max_display,
            plot_type="dot",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_summary_beeswarm.pdf"))
        plt.close()

        plot_results["summary"] = True
    except Exception as e:
        logging.error(f"Failed to create SHAP summary plots: {e}")
        plot_results["summary"] = False

    # 2. Pairwise dependence plots
    try:
        n_features = data_df.shape[1]
        if n_features <= 32:  # Only create if manageable number of features
            feature_pairs_done = set()
            for i in range(n_features):
                for j in range(i + 1, n_features):  # Only upper triangle
                    pair = tuple(sorted([feature_names.iloc[i], feature_names.iloc[j]]))
                    if pair in feature_pairs_done:
                        continue

                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(
                        i,
                        shap_values,
                        data_df,
                        interaction_index=j,
                        feature_names=feature_names.tolist(),
                        show=False,
                    )
                    plt.title(f"SHAP dependence: {pair[0]} vs {pair[1]}")
                    plt.tight_layout()

                    # Save using feature names in filename
                    filename = f"{_sanitize_filename(pair[0])}_vs_{_sanitize_filename(pair[1])}.pdf"
                    plt.savefig(dependence_dir.joinpath(filename))
                    plt.close()

                    feature_pairs_done.add(pair)

        plot_results["dependence"] = True
    except Exception as e:
        logging.error(f"Failed to create dependence plots: {e}")
        plot_results["dependence"] = False

    return plot_results


def create_interaction_plots(
    interaction_values: np.ndarray,
    feature_names: pd.Series,
    save_path: Path,
    prefix: str,
    max_display: int = 30,
) -> Dict[str, bool]:
    """Create visualization plots for SHAP interaction values.

    Args:
        interaction_values: SHAP interaction values [n_samples, n_features, n_features]
        feature_names: Names of features [n_features]
        save_path: Directory to save plots
        prefix: Prefix for plot filenames
        max_display: Maximum number of features to display

    Returns:
        Dictionary mapping plot types to success status:
            - 'matrix': Interaction strength matrix heatmap
            - 'ranking': Bar plot of strongest pairwise interactions

    Notes:
        - Uses absolute values for interaction strengths
        - Matrix shows top features based on total interaction strength
        - Bar plot shows strongest individual pairwise interactions
        - Colors use Blues colormap from white (weak) to dark blue (strong)
    """
    plot_results = {}

    # Average interaction values across samples and get absolute values
    mean_interactions = np.abs(np.mean(interaction_values, axis=0))

    # Make interaction matrix symmetric (average both directions)
    mean_interactions = (mean_interactions + mean_interactions.T) / 2

    # Remove self-interactions
    np.fill_diagonal(mean_interactions, 0)

    # 1. Complete interaction matrix heatmap
    try:
        plt.figure(figsize=(12, 10))
        # Create custom colormap: white to blue for increasing absolute values
        im = plt.imshow(
            mean_interactions,
            cmap="Blues",  # Use Blues colormap
            norm=Normalize(
                vmin=0,  # Start from 0 since we're using absolute values
                vmax=np.nanmax(mean_interactions),
            ),
        )
        plt.colorbar(im, label="Mean |SHAP interaction value|")

        # Add all feature labels
        plt.xticks(
            range(len(feature_names)),
            feature_names.tolist(),
            rotation=45,
            ha="right",
        )
        plt.yticks(range(len(feature_names)), feature_names.tolist())

        plt.title("Feature Interaction Matrix")
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_interactions_full.pdf"))
        plt.close()

        # Also create a version with only top features if there are many
        if len(feature_names) > max_display:
            total_strength = np.sum(mean_interactions, axis=1)
            top_idx = np.argsort(total_strength)[-max_display:]

            plt.figure(figsize=(12, 10))
            im = plt.imshow(
                mean_interactions[top_idx][:, top_idx],
                cmap="Blues",  # Use same colormap for consistency
                norm=Normalize(
                    vmin=0,
                    vmax=np.nanmax(mean_interactions),  # Use same scale as full matrix
                ),
            )
            plt.colorbar(im, label="Mean |SHAP interaction value|")

            plt.xticks(
                range(len(top_idx)),
                feature_names.iloc[top_idx].tolist(),
                rotation=45,
                ha="right",
            )
            plt.yticks(range(len(top_idx)), feature_names.iloc[top_idx].tolist())

            plt.title(f"Top {max_display} Feature Interactions")
            plt.tight_layout()
            plt.savefig(save_path.joinpath(f"{prefix}_interactions_top.pdf"))
            plt.close()

        plot_results["matrix"] = True
    except Exception as e:
        logging.error(f"Failed to create interaction matrix plots: {e}")
        plot_results["matrix"] = False

    # 2. Top unique interaction pairs
    try:
        # Get unique pairs of interactions (upper triangle only)
        i_upper, j_upper = np.triu_indices_from(mean_interactions, k=1)
        values = mean_interactions[i_upper, j_upper]

        # Sort by strength and take top pairs
        n_show = min(max_display, len(values))
        top_indices = np.argsort(values)[-n_show:]

        # Create feature pairs
        pairs = [
            f"{feature_names.iloc[i_upper[idx]]} × {feature_names.iloc[j_upper[idx]]}"
            for idx in top_indices
        ]
        top_values = values[top_indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_values)), top_values)
        plt.yticks(range(len(top_values)), pairs)
        plt.xlabel("Mean |SHAP interaction value|")
        plt.title("Strongest Feature Interactions")
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_interactions_ranking.pdf"))
        plt.close()
        plot_results["ranking"] = True
    except Exception as e:
        logging.error(f"Failed to create interaction ranking plot: {e}")
        plot_results["ranking"] = False

    return plot_results


def save_model_results(
    shap_results: ShapResults,
    test_scores: Dict[int, Dict[str, float]],
    data_df: pd.DataFrame,
    feature_annotations: pd.DataFrame,
    results_path: Path,
    prefix: str,
) -> None:
    """Save model results including SHAP values and performance metrics.

    Args:
        shap_results: Container with SHAP values and interaction values
        test_scores: Dict mapping random seeds to metric scores
        data_df: Original feature matrix [n_samples, n_features]
        feature_annotations: DataFrame with gene annotations
        results_path: Directory to save results
        prefix: Prefix for output filenames
    """
    # Ensure indices match by converting both to strings
    if feature_annotations is not None:
        feature_annotations.index = feature_annotations.index.astype(str)
    data_df_columns = data_df.columns.astype(str)

    # Get annotations only for features present in data
    valid_annotations = feature_annotations.loc[
        feature_annotations.index.intersection(data_df_columns)
    ]
    feature_names = valid_annotations["SYMBOL"]

    # Save SHAP values
    base_fname = f"{prefix}_shap"

    # First order SHAP values
    pd.DataFrame(
        shap_results.values,
        index=data_df.index,
        columns=data_df.columns.astype(str),
    ).to_csv(results_path.joinpath(f"{base_fname}_values.csv"))

    # Save feature-level summary statistics
    summary_stats = pd.DataFrame(
        {
            "mean_abs_shap": np.abs(shap_results.values).mean(axis=0),
            "std_abs_shap": np.abs(shap_results.values).std(axis=0),
            "mean_shap": shap_results.values.mean(axis=0),
            "std_shap": shap_results.values.std(axis=0),
        },
        index=data_df.columns.astype(str),
    )

    summary_df = pd.concat([summary_stats, feature_annotations], axis=1)
    summary_df.sort_values("mean_abs_shap", ascending=False).to_csv(
        results_path.joinpath(f"{base_fname}_summary.csv")
    )

    # Save filtered features at different thresholds
    for shap_thr in (1e-03, 1e-04, 1e-05):
        filtered_df = summary_df[summary_df["mean_abs_shap"] > shap_thr]
        if not filtered_df.empty:
            shap_thr_str = str(shap_thr).replace(".", "_")
            filtered_df.to_csv(
                results_path.joinpath(f"{base_fname}_filtered_{shap_thr_str}.csv")
            )

    # Save interaction values
    np.save(
        results_path.joinpath(f"{base_fname}_interaction_values.npy"),
        shap_results.interaction_values,
    )

    # Save interaction summary as DataFrame
    pd.DataFrame(
        shap_results.interaction_values.mean(axis=0),
        index=feature_names,
        columns=feature_names,
    ).to_csv(results_path.joinpath(f"{base_fname}_interaction_summary.csv"))

    # Save performance metrics
    pd.DataFrame(test_scores).transpose().sort_values(
        ["balanced_accuracy", "precision", "recall", "f1"], ascending=False
    ).to_csv(results_path.joinpath(f"{prefix}_test_scores.csv"))


def bootstrap_training(
    data: Union[Path, pd.DataFrame],
    annot_df: pd.DataFrame,
    contrast_factor: str,
    org_db: OrgDB,
    classifier_name: str,
    hparams_file: Path,
    results_path: Path,
    custom_features: Optional[pd.DataFrame] = None,
    custom_features_gene_type: Optional[str] = "ENTREZID",
    exclude_features: Optional[Iterable[str]] = None,
    n_jobs: int = 1,
    bootstrap_iterations: int = 1000,
    random_seed: int = 8080,
) -> None:
    """Bootstrap training to assess model stability and feature importance.

    Args:
        data: Gene expression matrix [n_features, n_samples] as file or DataFrame
        annot_df: Sample annotations with class labels
        contrast_factor: Column name in annot_df containing class labels
        org_db: Organism annotation database
        classifier_name: Name of the classifier to use
        hparams_file: JSON file containing best parameters from tuning
        results_path: Directory to save results
        custom_features: DataFrame with gene annotations having ENTREZID as index
                       and SYMBOL, GENENAME, GENETYPE as columns
        exclude_features: Features to remove from data
        n_jobs: Number of parallel jobs for supported classifiers and shapiq
        bootstrap_iterations: Number of bootstrap iterations
        random_seed: Random seed for reproducibility

    Notes:
        n_jobs parameter affects:
        - Random Forest classifier
        - LightGBM classifier
        - XGBoost classifier
        - SHAP interaction computation (shapiq)
    """
    # 0. Set seeds to ensure reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    ####################################################################################
    # 1. Data
    # 1.1. Get pre-processed data, class labels and overlapping features
    data_df, class_labels, overlapping_features, _ = process_gene_count_data(
        counts=data,
        annot_df=annot_df,
        contrast_factor=contrast_factor,
        org_db=org_db,
        custom_features=custom_features,
        custom_features_gene_type=custom_features_gene_type,
        exclude_features=exclude_features,
    )
    _validate_binary_classification(class_labels)

    if overlapping_features.empty:
        logging.warning(
            "There aren't any overlapping features, model training cancelled."
        )
        return

    ####################################################################################
    # 2. Model training
    # Define model - Hyper-parameters extracted from grid-search tuning
    with hparams_file.open("r") as fh:
        params = json.load(fh)
        # Update n_jobs in loaded parameters if not already present
        if classifier_name in ["random_forest", "light_gbm", "xgboost"]:
            params.setdefault("n_jobs", n_jobs)

    test_scores, shap_results = bootstrap_relevant_features(
        counts_df=data_df,
        class_labels=class_labels,
        classifier_name=classifier_name,
        model_params=params,
        random_seeds=bootstrap_iterations,
        n_jobs=n_jobs,
    )

    ####################################################################################
    # 3. Process and save results
    results_path.mkdir(exist_ok=True, parents=True)

    # Ensure indices match
    if custom_features is not None:
        custom_features.index = custom_features.index.astype(str)
    data_columns = data_df.columns.astype(str)

    if custom_features is not None:
        feature_annotations = custom_features.loc[
            custom_features.index.intersection(data_columns)
        ]
    else:
        feature_annotations = pd.DataFrame(index=data_columns)
        feature_annotations["SYMBOL"] = data_columns

    # Check if first order SHAP values need transposing
    if shap_results.values.shape[0] != len(data_df.index):
        shap_results.values = shap_results.values.T

    # 3.1. Save all results
    prefix = f"bootstrap_{bootstrap_iterations}"
    save_model_results(
        shap_results=shap_results,
        test_scores=test_scores,
        data_df=data_df,
        feature_annotations=feature_annotations,
        results_path=results_path,
        prefix=prefix,
    )

    # 3.2. Create visualizations
    # First order SHAP plots
    try:
        shap_plots = create_shap_plots(
            shap_values=shap_results.values,
            data_df=data_df,
            feature_names=feature_annotations["SYMBOL"],
            save_path=results_path,
            prefix=f"{prefix}_shap",
            max_display=30,
        )
        logging.info(
            "Successfully created "
            f"{sum(shap_plots.values())}/{len(shap_plots)} "
            "first-order SHAP plots"
        )
    except Exception as e:
        logging.error(f"Failed to create first-order SHAP visualizations: {e}")

    # Second order SHAP plots
    try:
        shap_interactions_plots = create_interaction_plots(
            interaction_values=shap_results.interaction_values,
            feature_names=feature_annotations["SYMBOL"],
            save_path=results_path,
            prefix=f"{prefix}_shap_interactions",
            max_display=30,
        )
        logging.info(
            "Successfully created "
            f"{sum(shap_interactions_plots.values())}/{len(shap_interactions_plots)} "
            "second-order SHAP plots"
        )
    except Exception as e:
        logging.error(f"Failed to create second-order SHAP visualizations: {e}")
