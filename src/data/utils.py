import contextlib
import os
import signal
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain, combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from r_wrappers.utils import map_gene_id


def gene_expression_levels(
    expr_df: pd.DataFrame,
    gene_expr_col: str,
    gene_expr_level: str,
    percentile: int = 10,
) -> pd.DataFrame:
    """Classify gene expression values into low, mid, and high levels.

    Args:
        expr_df: Contains a column with the log counts of a given gene
        gene_expr_col: Column containing expression values
        gene_expr_level: Column name for the new column containing expression levels
        percentile: Percentile threshold for low/high classification (default: 10)
            Values below percentile -> "low"
            Values above (100-percentile) -> "high"
            Values in between -> "mid"

    Returns:
        DataFrame with new column containing expression level labels
    """
    # 0. Get user-provided percentiles
    expr_df = deepcopy(expr_df)
    p0, p1 = np.percentile(expr_df[gene_expr_col], (percentile, 100 - percentile))

    # 1. Classify data into three groups
    expr_df.loc[expr_df[gene_expr_col] < p0, gene_expr_level] = "low"
    expr_df.loc[
        (expr_df[gene_expr_col] >= p0) & (expr_df[gene_expr_col] <= p1), gene_expr_level
    ] = "mid"
    expr_df.loc[expr_df[gene_expr_col] > p1, gene_expr_level] = "high"

    return expr_df


def run_cmd(
    cmd: Iterable[str], log_path: Optional[Path] = None
) -> Optional[Dict[str, str]]:
    """Run console command with optional logging.

    Args:
        cmd: Command components as strings, e.g. ['/bin/prog', '-i', 'data.txt']
        log_path: Optional path to store stdout and stderr logs

    Returns:
        Dict with 'stdout' and 'stderr' if successful, None otherwise
    """
    # 0. Run commands separated by pipes
    process_output = None
    try:
        for i, process in enumerate(" ".join([str(x) for x in cmd]).split("|")):
            process_input = (
                process_output.stdout if process_output is not None else None
            )
            process_output = subprocess.run(
                process.strip().split(" "),
                input=process_input,
                check=True,
                capture_output=True,
                universal_newlines=True,
            )
    except subprocess.CalledProcessError as e:
        print(e)

    # 1. Save stdout and stderr if command execution was successful and a
    # file path is provided
    if process_output is not None:
        logs = {
            "stderr": str(process_output.stderr),
            "stdout": str(process_output.stdout),
        }
        if log_path is not None and log_path.suffix == ".log":
            log_path.write_text(
                f"stderr:\n {logs['stderr']} \n\n stdout:\n {logs['stdout']}"
            )

        return logs


def parallelize_star(
    func: Callable[..., Any],
    inputs: Iterable[Tuple[Any, ...]],
    threads: int = 8,
    method: str = "spawn",
) -> list:
    """Parallel execution using starmap.

    Args:
        func: Function to parallelize
        inputs: Iterable of argument tuples for each function call
        threads: Number of parallel processes
        method: Multiprocessing start method ('spawn' or 'fork')

    Returns:
        List of results from parallel function calls
    """
    with get_context(method).Pool(threads, maxtasksperchild=1) as pool:
        return pool.starmap(func, tqdm(inputs))


def parallelize_map(
    func: Callable[[Any], Any],
    inputs: Iterable[Any],
    threads: int = 8,
    method: str = "spawn",
) -> list:
    """Parallel execution using imap_unordered.

    Args:
        func: Function to parallelize that takes single argument
        inputs: Iterable of arguments for function calls
        threads: Number of parallel processes
        method: Multiprocessing start method ('spawn' or 'fork')

    Returns:
        List of results from parallel function calls
    """
    with get_context(method).Pool(threads, maxtasksperchild=1) as pool:
        return list(tqdm(pool.imap_unordered(func, inputs), total=len(inputs)))


def filter_genes_wrt_annotation(
    genes: Iterable[str], org_db: OrgDB, from_type: str = "ENSEMBL"
) -> list[str]:
    """Filter genes based on annotation database criteria.

    Args:
        genes: Gene IDs to filter
        org_db: Organism annotation database for ID mapping and filtering
        from_type: Input gene ID type ('ENSEMBL', 'ENTREZID', etc.)

    Returns:
        List of filtered gene IDs that:
        - Have valid ENTREZID mappings
        - Are protein-coding or ncRNA genes
        - Have unique mappings (no duplicates)
    """
    # 1. Filter genes without ENTREZID
    genes_entrezid = map_gene_id(genes, org_db, from_type, "ENTREZID")
    genes_entrezid = (
        genes_entrezid[~genes_entrezid.str.contains("/", na=False)]
        .dropna()
        .drop_duplicates(keep=False)
    )

    # 2. Remove genes with unwanted characteristics
    gene_types = map_gene_id(genes_entrezid.index, org_db, from_type, "GENETYPE")
    gene_types = gene_types[~genes_entrezid.str.contains("/", na=False)].dropna()[
        gene_types.isin(["protein-coding", "ncRNA"])
    ]

    return gene_types.index.tolist()


def filter_df(
    df: pd.DataFrame, filter_values: Dict[str, Iterable[Any]]
) -> pd.DataFrame:
    """Filter DataFrame rows by matching values in columns.

    Args:
        df: DataFrame to filter
        filter_values: Dict mapping column names to allowed values

    Returns:
        Filtered DataFrame containing only rows where column values match targets

    Raises:
        AssertionError: If any filter column doesn't exist in DataFrame
    """
    # ensure that all fields are valid
    assert all([k in df.columns for k in filter_values.keys()])

    # filter dataframe
    return df[
        np.logical_and.reduce(
            [
                df[column].isin(target_values)
                for column, target_values in filter_values.items()
            ]
        )
    ]


def select_data_classes(
    metadata: pd.DataFrame, classes_filters: Iterable[Dict[str, Iterable[Any]]]
) -> list[Iterable[Any]]:
    """Select non-overlapping sample classes using filters.

    Args:
        metadata: Sample metadata with samples as index
        classes_filters: List of filter dicts, one per class. Each dict maps
            column names to allowed values for that class

    Returns:
        List of sample ID iterables, one per class, with no overlap between classes

    Raises:
        AssertionError: If any samples appear in multiple classes
    """
    # 1. Filter dataframe and get sample IDs for each class
    class_samples_ids = [
        filter_df(metadata, classes_filters).index
        for classes_filters in classes_filters
    ]

    # 2. Check that the samples of the different classes do not intersect
    assert len(set(metadata.index).intersection(*class_samples_ids)) == 0, (
        "There are overlapping samples among classes, please check the class filters"
    )

    # 3. Return class ids
    return class_samples_ids


def ranges_overlap(ranges: Iterable[Tuple[float, float]]) -> bool:
    """Check if numeric ranges have a common intersection point.

    Args:
        ranges: Iterable of (min, max) tuples defining numeric ranges

    Returns:
        True if there exists a value present in all ranges,
        False if ranges have no common intersection

    Example:
        >>> ranges_overlap([(0,5), (3,8), (4,10)])
        True  # 4 is in all ranges
        >>> ranges_overlap([(0,3), (4,8)])
        False  # no value in both ranges
    """
    # Find the maximum of all minimums and minimum of all maximums
    max_min = max(min_val for min_val, _ in ranges)
    min_max = min(max_val for _, max_val in ranges)

    # If max_min <= min_max, there exists a point contained in all ranges
    return max_min <= min_max


def get_overlapping_features(
    data_df: pd.DataFrame,
    class_samples_ids: Iterable[Iterable[Any]],
    class_names: Optional[Iterable[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Identify features with overlapping value ranges across classes.

    Args:
        data_df: Feature matrix [n_samples, n_features]
        class_samples_ids: Iterable of sample ID iterables, one per class
        class_names: Optional names for each class (if None, uses 0..N-1)

    Returns:
        Tuple containing:
        - Boolean Series indicating which features overlap across all classes
        - DataFrame with min/max ranges per feature per class with class names as columns
    """
    data_df_class = deepcopy(data_df)

    # 1. Assign class IDs/names to samples
    class_labels = range(len(class_samples_ids)) if class_names is None else class_names
    for label, samples_ids in zip(class_labels, class_samples_ids):
        data_df_class.loc[samples_ids, "class"] = label

    # 2. Get min and max values per class
    data_df_ranges = data_df_class.groupby("class").agg(("min", "max")).transpose()

    # 3. For each feature, check if its ranges overlap across ALL classes
    return (
        data_df_ranges.groupby(level=0).apply(
            lambda feature: ranges_overlap([x[1] for x in feature.items()])
        ),
        data_df_ranges,
    )


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager that raises TimeoutException after specified seconds.

    Args:
        seconds: Number of seconds before timeout

    Raises:
        TimeoutException: If execution time exceeds specified seconds
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def powerset(iterable: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    """Generate all possible combinations of elements.

    Args:
        iterable: Input elements

    Returns:
        Iterator yielding tuples for all possible element combinations,
        from empty tuple to full set
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def supress_stdout(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to suppress stdout during function execution.

    Args:
        func: Function whose stdout should be suppressed

    Returns:
        Wrapped function that executes with suppressed stdout
    """

    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper
