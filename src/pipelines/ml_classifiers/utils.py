import json
import logging
import pickle
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import shap
import shap.plots as shap_plots
import shapiq
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
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
from r_wrappers.complex_heatmaps import complex_heatmap, heatmap_annotation

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
    """Create and configure a classifier instance.

    Args:
        classifier_name: Name of classifier to create
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs for supported classifiers
        **kwargs: Additional parameters to pass to classifier

    Returns:
        Configured classifier instance

    Raises:
        ValueError: If classifier_name is not recognized
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = random_seed
    if "n_jobs" not in kwargs and classifier_name in [
        "random_forest",
        "light_gbm",
        "xgboost",
    ]:
        kwargs["n_jobs"] = n_jobs

    if classifier_name == "decision_tree":
        return DecisionTreeClassifier(**kwargs)
    elif classifier_name == "random_forest":
        return RandomForestClassifier(**kwargs)
    elif classifier_name == "light_gbm":
        return LGBMClassifier(
            device="cpu",
            verbose=-1,
            min_gain_to_split=0,
            **kwargs,
        )
    elif classifier_name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            **kwargs,
        )
    elif classifier_name == "nu_svc":
        return NuSVC(**kwargs)
    elif classifier_name == "mlp":
        return MLPClassifier(**kwargs)
    elif classifier_name == "tabpfn":
        return TabPFNClassifier(device="cpu", **kwargs)
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
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


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
    data_df, class_labels, overlapping_features, data_df_ranges, _ = (
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
    non_overlapping_features = overlapping_features[~overlapping_features]
    if not non_overlapping_features.empty:
        non_overlapping_features_df = custom_features[
            custom_features.index.isin(non_overlapping_features.index.astype(str))
        ]

        counts_matrix = pd.read_csv(data, index_col=0).loc[
            non_overlapping_features_df.index,
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
            save_path=results_path.joinpath("non_overlapping_features_clustering.pdf"),
            width=10,
            height=10,
            name="Non-overlapping genes",
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


def bootstrap_relevant_features(
    counts_df: pd.DataFrame,
    class_labels: np.ndarray,
    model: Union[
        RandomForestClassifier,
        DecisionTreeClassifier,
        XGBClassifier,
        LGBMClassifier,
        NuSVC,
        MLPClassifier,
        TabPFNClassifier,
    ],
    random_seeds: Union[int, Iterable[int]] = 100,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, np.ndarray]]:
    """Train a model multiple times to compute feature importance scores.

    Args:
        counts_df: Input feature matrix [n_samples, n_features]
        class_labels: Binary class labels [n_samples]
        model: Pre-configured classifier instance for binary classification
        random_seeds: Number of iterations or list of random seeds to use

    Returns:
        Tuple containing:
        - Dict mapping seeds to performance metrics
        - Dict mapping interaction order (1-5) to averaged arrays with shapes:
            - order 1: [n_samples, n_features]
            - order 2: [n_samples, n_features, n_features]
            - order 3: [n_samples, n_features, n_features, n_features]
            etc.

    Note:
        Arrays are averaged across bootstrap iterations but maintain sample dimension
    """
    # 0. Setup
    test_scores = defaultdict(dict)
    n_samples, n_features = counts_df.shape

    # Initialize mean arrays for orders 1-5
    shap_interactions_mean = {
        order: np.zeros([n_samples] + [n_features] * order) for order in range(1, 6)
    }
    n_successful = 0

    # Add warning filter within the function as well
    warnings.filterwarnings(
        "ignore", message="No further splits with positive gain", category=UserWarning
    )

    # 1. Choose random_seeds
    random_seeds = (
        random_seeds
        if isinstance(random_seeds, list)
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
        test_scores[random_seed].update(get_model_metrics(test_labels, test_pred))

        # 2.5. Get SHAP values and interactions
        try:
            explainer = shapiq.Explainer(
                model=model,
                data=np.ascontiguousarray(test_data),
                index="FSII",
            )

            # Process each order (1-5)
            for order in range(1, 6):
                try:
                    # Get SHAP values for this iteration
                    interactions = np.stack(
                        [
                            explainer.explain(sample).get_n_order_values(order)
                            for sample in np.ascontiguousarray(counts_df)
                        ]
                    )

                    # Update running average
                    if n_successful == 0:
                        shap_interactions_mean[order] = interactions
                    else:
                        shap_interactions_mean[order] = (
                            shap_interactions_mean[order] * n_successful + interactions
                        ) / (n_successful + 1)
                except Exception as e:
                    logging.warning(f"Order-{order} SHAP interactions failed: {e}")

            n_successful += 1

        except Exception as e:
            logging.warning(f"SHAP computation failed: {e}")

    return test_scores, shap_interactions_mean


def plot_interaction_heatmap(
    interaction_values: np.ndarray,
    feature_names: pd.Index,
    max_features: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> None:
    """Plot heatmap visualization of pairwise SHAP interaction values.

    Args:
        interaction_values: Second order SHAP values [n_samples, n_features, n_features]
        feature_names: Names corresponding to each feature index [n_features]
        max_features: Maximum number of top features to display
        figsize: Figure dimensions (width, height) in inches
        save_path: Optional path to save plot file

    Notes:
        - Features are selected based on total absolute interaction strength
        - Uses RdBu colormap centered at 0
        - If 3D array is provided, averages across samples first
    """
    # Average interactions across samples
    mean_interactions = np.mean(interaction_values, axis=0)

    # Get top features based on total interaction strength
    total_interactions = np.sum(np.abs(mean_interactions), axis=(0, 1))
    top_idx = np.argsort(total_interactions)[-max_features:]

    # Create heatmap data
    heatmap_data = pd.DataFrame(
        mean_interactions[top_idx][:, top_idx],
        index=feature_names[top_idx],
        columns=feature_names[top_idx],
    )

    plt.figure(figsize=figsize)
    plt.imshow(heatmap_data, cmap="RdBu")
    plt.colorbar(label="SHAP interaction value")

    # Add labels
    plt.xticks(
        range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45, ha="right"
    )
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

    plt.title("SHAP Feature Interactions")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def create_first_order_shap_plots(
    shap_values: np.ndarray,
    data_df: pd.DataFrame,
    feature_names: pd.Index,
    save_path: Path,
    prefix: str,
    max_display: int = 30,
) -> Dict[str, bool]:
    """Create various SHAP visualization plots with error handling.

    Args:
        shap_values: First-order SHAP values [n_samples, n_features]
        data_df: Original feature matrix [n_samples, n_features]
        feature_names: Names of features [n_features]
        save_path: Directory to save plots
        prefix: Prefix for plot filenames
        max_display: Maximum number of features to display

    Returns:
        Dictionary mapping plot types to success status:
        - 'bar': Feature importance bar plot
        - 'beeswarm': SHAP value distributions
        - 'dependence': Feature dependence plots (top 5)
        - 'force_summary': Aggregated force plot
        - 'interaction_overview': Feature interaction dot plot
    """
    plot_results = {}

    # 1. Bar plot
    try:
        shap.summary_plot(
            shap_values,
        )
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_bar.pdf"))
        plt.close()
        plot_results["bar"] = True
    except Exception as e:
        logging.error(f"Failed to create bar plot: {e}")
        plot_results["bar"] = False

    # 2. Beeswarm plot
    try:
        shap.summary_plot(
            shap_values,
            data_df,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_beeswarm.pdf"))
        plt.close()
        plot_results["beeswarm"] = True
    except Exception as e:
        logging.error(f"Failed to create beeswarm plot: {e}")
        plot_results["beeswarm"] = False

    # 3. Dependence plots for top features
    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-5:]  # Top 5 features

        for idx in top_features_idx:
            feature_name = feature_names[idx]
            shap.dependence_plot(
                idx,
                shap_values,
                data_df,
                feature_names=feature_names,
                show=False,
            )
            plt.title(f"Dependence plot for {feature_name}")
            plt.tight_layout()
            plt.savefig(save_path.joinpath(f"{prefix}_dependence_{feature_name}.pdf"))
            plt.close()
        plot_results["dependence"] = True
    except Exception as e:
        logging.error(f"Failed to create dependence plots: {e}")
        plot_results["dependence"] = False

    # 4. Summary force plot
    try:
        shap_plots.force(
            shap.Explanation(
                values=shap_values.mean(axis=0),
                base_values=np.zeros(1),
                data=data_df.mean(axis=0),
                feature_names=feature_names,
            ),
            show=False,
            matplotlib=True,
        )
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_force_summary.pdf"))
        plt.close()
        plot_results["force_summary"] = True
    except Exception as e:
        logging.error(f"Failed to create force summary plot: {e}")
        plot_results["force_summary"] = False

    # 5. Interaction overview heatmap
    try:
        shap.summary_plot(
            shap_values,
            data_df,
            feature_names=feature_names,
            plot_type="compact_dot",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_interaction_overview.pdf"))
        plt.close()
        plot_results["interaction_overview"] = True
    except Exception as e:
        logging.error(f"Failed to create interaction overview plot: {e}")
        plot_results["interaction_overview"] = False

    return plot_results


def create_second_order_shap_plots(
    interaction_values: np.ndarray,
    feature_names: pd.Index,
    save_path: Path,
    prefix: str,
    max_display: int = 30,
) -> Dict[str, bool]:
    """Create plots for second-order SHAP interaction values.

    Args:
        interaction_values: Second order SHAP values [n_samples, n_features, n_features]
        feature_names: Names of features [n_features]
        save_path: Directory to save plots
        prefix: Prefix for plot filenames
        max_display: Maximum number of features to display

    Returns:
        Dictionary mapping plot types to success status:
        - 'interaction_heatmap': Pairwise interaction strength heatmap
        - 'interaction_bar': Top features by total interaction strength
    """
    plot_results = {}

    # 1. Standard interaction heatmap
    try:
        plot_interaction_heatmap(
            interaction_values,
            feature_names,
            max_features=max_display,
            save_path=save_path.joinpath(f"{prefix}_interaction_heatmap.pdf"),
        )
        plot_results["interaction_heatmap"] = True
    except Exception as e:
        logging.error(f"Failed to create interaction heatmap: {e}")
        plot_results["interaction_heatmap"] = False

    # 2. Top interactions bar plot
    try:
        # Get top interactions
        abs_interactions = np.abs(interaction_values)
        total_effect = np.sum(abs_interactions, axis=1)
        top_idx = np.argsort(total_effect)[-max_display:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_idx)), total_effect[top_idx])
        plt.yticks(range(len(top_idx)), feature_names[top_idx])
        plt.xlabel("Total Interaction Strength")
        plt.title("Top Feature Interaction Effects")
        plt.tight_layout()
        plt.savefig(save_path.joinpath(f"{prefix}_top_interactions_bar.pdf"))
        plt.close()
        plot_results["interaction_bar"] = True
    except Exception as e:
        logging.error(f"Failed to create top interactions bar plot: {e}")
        plot_results["interaction_bar"] = False

    return plot_results


def save_model_results(
    shap_interactions: Dict[int, np.ndarray],
    test_scores: Dict[int, Dict[str, float]],
    data_df: pd.DataFrame,
    feature_annotations: pd.DataFrame,
    results_path: Path,
    prefix: str,
) -> None:
    """Save all model results including SHAP values and performance metrics.

    Args:
        shap_interactions: Dict mapping order (1-5) to arrays with shapes:
            - order 1: [n_samples, n_features]
            - order 2: [n_samples, n_features, n_features]
            - order 3: [n_samples, n_features, n_features, n_features]
            etc.
        test_scores: Dict mapping random seeds to metric scores
        data_df: Original feature matrix [n_samples, n_features]
        feature_annotations: DataFrame with gene annotations
        results_path: Directory to save results
        prefix: Prefix for output filenames

    Outputs:
        Per-sample values and summary statistics for each interaction order
        See function docstring for complete file list
    """
    feature_names = feature_annotations["SYMBOL"]

    # Save all interaction orders
    for order, interaction_iterations in shap_interactions.items():
        base_fname = f"{prefix}_shap_interactions_order_{order}"

        if order == 1:  # First order SHAP values
            # Average across bootstrap iterations but keep sample dimension
            shap_values = np.mean(
                interaction_iterations, axis=0
            )  # [n_samples, n_features]

            # Save raw per-sample SHAP values
            pd.DataFrame(
                shap_values,
                index=data_df.index,
                columns=data_df.columns.astype(str),
            ).to_csv(results_path.joinpath(f"{base_fname}_per_sample.csv"))

            # Save feature-level summary statistics
            summary_stats = pd.DataFrame(
                {
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                    "std_abs_shap": np.abs(shap_values).std(axis=0),
                    "mean_shap": shap_values.mean(axis=0),
                    "std_shap": shap_values.std(axis=0),
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
                        results_path.joinpath(
                            f"{base_fname}_filtered_{shap_thr_str}.csv"
                        )
                    )

        else:  # Higher order interactions
            # Average across bootstrap iterations
            interactions_mean = np.mean(interaction_iterations, axis=0)

            if order == 2:  # Second order gets special treatment
                pd.DataFrame(
                    interactions_mean.mean(axis=0),  # Average across samples
                    index=feature_names,
                    columns=feature_names,
                ).to_csv(results_path.joinpath(f"{base_fname}.csv"))

            # Save raw arrays for all orders
            np.save(results_path.joinpath(f"{base_fname}.npy"), interactions_mean)

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
    data_df, class_labels, overlapping_features, _, label_encoder = (
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
        # Update n_jobs in loaded parameters
        if classifier_name in ["random_forest", "light_gbm", "xgboost"]:
            params["n_jobs"] = n_jobs

        classifier = get_classifier(classifier_name, random_seed, n_jobs, **params)

    test_scores, shap_interactions = bootstrap_relevant_features(
        counts_df=data_df,
        class_labels=class_labels,
        model=classifier,
        random_seeds=bootstrap_iterations,
    )

    ####################################################################################
    # 3. Process and save results
    results_path.mkdir(exist_ok=True, parents=True)

    # 3.1. Save all results
    prefix = f"bootstrap_{bootstrap_iterations}"
    save_model_results(
        shap_interactions=shap_interactions,
        test_scores=test_scores,
        data_df=data_df,
        feature_annotations=custom_features.loc[data_df.columns.astype(str)],
        results_path=results_path,
        prefix=prefix,
    )

    # 3.2. Create visualizations
    feature_names = custom_features.loc[data_df.columns.astype(str)]["SYMBOL"]

    # First order SHAP plots
    try:
        first_order_plots = create_first_order_shap_plots(
            shap_values=shap_interactions[1],
            data_df=data_df,
            feature_names=feature_names,
            save_path=results_path,
            prefix=f"{prefix}_first_order",
            max_display=30,
        )
        logging.info(
            f"Successfully created {sum(first_order_plots.values())}/{len(first_order_plots)} "
            "first-order SHAP plots"
        )
    except Exception as e:
        logging.error(f"Failed to create first-order SHAP visualizations: {e}")

    # Second order SHAP plots
    try:
        second_order_plots = create_second_order_shap_plots(
            interaction_values=shap_interactions[2],
            feature_names=feature_names,
            save_path=results_path,
            prefix=f"{prefix}_second_order",
            max_display=30,
        )
        logging.info(
            f"Successfully created {sum(second_order_plots.values())}/{len(second_order_plots)} "
            "second-order SHAP plots"
        )
    except Exception as e:
        logging.error(f"Failed to create second-order SHAP visualizations: {e}")
