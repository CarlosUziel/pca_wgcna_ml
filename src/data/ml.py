import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn import preprocessing

from components.functional_analysis.orgdb import OrgDB
from data.utils import filter_genes_wrt_annotation, get_overlapping_features
from r_wrappers.utils import map_gene_id


def process_gene_count_data(
    counts_file: Path,
    annot_df: Path,
    contrast_factor: str,
    org_db: OrgDB,
    custom_genes_file: Path = None,
    exclude_genes: Iterable[str] = None,
) -> Tuple[pd.DataFrame, Iterable[int], Iterable[str], pd.DataFrame]:
    """Perform multiple pre-processing on gene expression data.

    Args:
        counts_file: A .csv file containing expression data of shape
            [n_genes, n_samples]
        annot_df: A pandas Dataframe containing samples annotations.
        contrast_factor: Column name containing the classes used for classification.
        org_db: Organism annotation database.
        custom_genes_path: A .csv file where the first column is a list of relevant
            ENSEMBL IDs genes, such as DEGs.
        exclude_genes: ENSEMBL ID genes to remove from data.

    Returns:
        Processed data.
    """
    # 1. Data
    # 1.1. Loading
    counts_df = pd.read_csv(counts_file, index_col=0).transpose()

    assert not counts_df.empty and not annot_df.empty, (
        "Counts or annotation dataframes are empty."
    )

    # 1.2. Only keep common samples between data and sample annotation
    common_idxs = list(set(counts_df.index).intersection(set(annot_df.index)))
    counts_df = counts_df.loc[common_idxs, :]
    annot_df = annot_df.loc[common_idxs, :]

    assert len(set(annot_df[contrast_factor])) == 2, (
        "Classes were lost after unifying count and annotation data. Please check input"
        " data."
    )

    # 1.3. Build class labels
    label_encoder = preprocessing.LabelEncoder()
    class_labels = label_encoder.fit_transform(annot_df[contrast_factor])

    # 2. Select genes
    if custom_genes_file is not None:
        custom_genes_df = pd.read_csv(custom_genes_file, index_col=0)

        genes = (
            custom_genes_df.index if not custom_genes_df.empty else counts_df.columns
        )

    else:
        genes = counts_df.columns

    if exclude_genes:
        genes = [gene for gene in genes if gene not in exclude_genes]

    # 2.1. Filter genes based on IDs and names
    genes = filter_genes_wrt_annotation(genes, org_db, "ENSEMBL")

    try:
        counts_df = counts_df.loc[:, genes]
    except KeyError:
        logging.warning(
            "Some of the input genes were not found in data_df and are thus ignored."
        )
        counts_df = counts_df.loc[:, counts_df.columns.intersection(set(genes))]

    # 2.2. Get ENTREZID genes IDs
    counts_df.columns = map_gene_id(counts_df.columns, org_db, "ENSEMBL", "ENTREZID")
    # get rid of non-uniquely mapped transcripts
    counts_df = counts_df.loc[:, ~counts_df.columns.str.contains("/", na=False)]
    # remove all transcripts that share ENTREZIDs IDs
    counts_df = counts_df.loc[:, counts_df.columns.dropna().drop_duplicates(keep=False)]

    # 3. Remove non-overlapping genes
    overlapping_genes, counts_df_ranges = get_overlapping_features(
        counts_df,
        [
            annot_df[annot_df[contrast_factor] == class_label].index
            for class_label in label_encoder.classes_
        ],
    )

    counts_df = counts_df.loc[:, overlapping_genes]

    # 4. Scale feature (gene) values between 0 and 1.
    counts_df = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(counts_df),
        index=counts_df.index,
        columns=counts_df.columns,
    )

    return counts_df, class_labels, overlapping_genes, counts_df_ranges, label_encoder
