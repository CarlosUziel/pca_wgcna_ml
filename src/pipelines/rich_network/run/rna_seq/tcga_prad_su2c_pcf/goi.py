import argparse
import functools
import json
import logging
import multiprocessing
import warnings
from itertools import chain, product
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.rich_network.utils import rich_wgcna_network
from r_wrappers.utils import map_gene_id
from utils import run_func_dict, trunc

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root-dir",
    type=str,
    help="Root directory",
    nargs="?",
    default="/media/ssd/Perez/storage",
)
parser.add_argument(
    "--threads",
    type=int,
    help="Number of threads for parallel processing",
    nargs="?",
    default=multiprocessing.cpu_count() - 2,
)

user_args = vars(parser.parse_args())
STORAGE: Path = Path(user_args["root_dir"])
GOI_ENSEMBL: str = "ENSG00000086205"  # ENSEMBL ID for FOLH1 (PSMA)
SPECIES: str = "Homo sapiens"
org_db = OrgDB(SPECIES)
GOI_SYMBOL = map_gene_id([GOI_ENSEMBL], org_db, "ENSEMBL", "SYMBOL")[0]
DATA_ROOT: Path = STORAGE.joinpath(f"TCGA_PRAD_SU2C_RNASeq_{GOI_SYMBOL}")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(
    f"samples_annotation_tcga_prad_su2c_clusters_{GOI_SYMBOL}_short.csv"
)
ANNOT_PATH: Path = DATA_PATH.joinpath(
    f"samples_annotation_tcga_prad_su2c_clusters_{GOI_SYMBOL}_short.csv"
)
COUNTS_PATH: Path = (
    STORAGE.joinpath("TCGA_PRAD_SU2C_RNASeq").joinpath("data").joinpath("star_counts")
)
SAMPLE_CONTRAST_FACTOR: str = "sample_cluster_no_replicates"
GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
PERCENTILES: Iterable[int] = (10, 15, 20)

########################################################################################
# Contrasts definition
########################################################################################
WITHIN_SAMPLES_CONTRASTS: Iterable[str] = ("prim", "met")
WITHIN_SAMPLE_CONTRASTS_FILTERS = [
    (
        {
            SAMPLE_CONTRAST_FACTOR: (sample_type,),
            f"{GOI_LEVEL_PREFIX}_{percentile}": ("high",),
        },
        {
            SAMPLE_CONTRAST_FACTOR: (sample_type,),
            f"{GOI_LEVEL_PREFIX}_{percentile}": ("low",),
        },
    )
    for sample_type in WITHIN_SAMPLES_CONTRASTS
    for percentile in PERCENTILES
    if not (sample_type == "met" and percentile != 20)
]
INTER_SAMPLE_CONTRASTS: Iterable[Tuple[Tuple[str, str]]] = (
    (("met", "high"), ("prim", "high")),
    (("met", "high"), ("prim", "low")),
    (("met", "low"), ("prim", "high")),
    (("met", "low"), ("prim", "low")),
)
INTER_SAMPLE_CONTRASTS_FILTERS = [
    (
        {
            SAMPLE_CONTRAST_FACTOR: (test[0],),
            f"{GOI_LEVEL_PREFIX}_{percentile_0}": (test[1],),
        },
        {
            SAMPLE_CONTRAST_FACTOR: (control[0],),
            f"{GOI_LEVEL_PREFIX}_{percentile_1}": (control[1],),
        },
    )
    for test, control in INTER_SAMPLE_CONTRASTS
    for percentile_0, percentile_1 in product(PERCENTILES, repeat=2)
    if not (
        (test[0] == "met" and percentile_0 != 20)
        or (control[0] == "met" and percentile_1 != 20)
    )
]
SIMPLE_SAMPLE_CONTRASTS: Iterable[Tuple[str, str]] = (
    ("prim", "norm"),
    ("met", "prim"),
)
SIMPLE_SAMPLE_CONTRASTS_FILTERS = [
    (
        {
            SAMPLE_CONTRAST_FACTOR: (test,),
        },
        {
            SAMPLE_CONTRAST_FACTOR: (control,),
        },
    )
    for test, control in SIMPLE_SAMPLE_CONTRASTS
]
ASYMETRIC_SAMPLE_CONTRASTS: Iterable[Tuple[Tuple[str, str]]] = (
    (("prim", "high"), ("norm", None)),
    (("prim", "low"), ("norm", None)),
    (("met", "high"), ("prim", None)),
    (("met", "low"), ("prim", None)),
)
ASYMETRIC_SAMPLE_CONTRASTS_FILTERS = [
    (
        {
            SAMPLE_CONTRAST_FACTOR: (test[0],),
            f"{GOI_LEVEL_PREFIX}_{percentile}": (test[1],),
        },
        {
            SAMPLE_CONTRAST_FACTOR: (control[0],),
        },
    )
    for test, control in ASYMETRIC_SAMPLE_CONTRASTS
    for percentile in PERCENTILES
]
SAMPLE_CLUSTER_CONTRAST_LEVELS: Iterable = (
    WITHIN_SAMPLE_CONTRASTS_FILTERS
    + INTER_SAMPLE_CONTRASTS_FILTERS
    + ASYMETRIC_SAMPLE_CONTRASTS_FILTERS
    + SIMPLE_SAMPLE_CONTRASTS_FILTERS
)
with DATA_ROOT.joinpath("SAMPLE_CLUSTER_CONTRAST_LEVELS.json").open("w") as fp:
    json.dump(SAMPLE_CLUSTER_CONTRAST_LEVELS, fp, indent=True)

########################################################################################
########################################################################################
########################################################################################

with DATA_ROOT.joinpath("SAMPLE_CLUSTER_CONTRAST_LEVELS.json").open("w") as fp:
    json.dump(SAMPLE_CLUSTER_CONTRAST_LEVELS, fp, indent=True)

P_COLS: List[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: List[str] = ["all", "up", "down"]
LFC_THS: Iterable[float] = (1.0,)
SPECIES: str = "Homo sapiens"
NETWORK_TYPES: List[str] = ("signed",)
CORRELATION_TYPES: List[str] = ("bicor",)
ITERATIVE_MODES: Iterable[bool] = (True, False)
CLASSIFIER_NAMES: List[str] = ("decision_tree", "random_forest")
BOOTSTRAP_ITERATIONS: Iterable[int] = (10000,)
CORR_THS: Iterable[float] = trunc(np.arange(0.1, 1, 0.1), decs=1)
REMOVE_ISOLATED: Iterable[bool] = (True, False)
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

input_collection = []
for (
    sample_cluster_contrast,
    percentile,
    p_col,
    p_th,
    lfc_level,
    lfc_th,
    network_type,
    correlation_type,
    iterative,
    classifier_name,
    bootstrap_iterations,
    corr_th,
    remove_isolated,
) in product(
    SAMPLE_CLUSTER_CONTRAST_LEVELS,
    PERCENTILES,
    P_COLS,
    P_THS,
    LFC_LEVELS,
    LFC_THS,
    NETWORK_TYPES,
    CORRELATION_TYPES,
    ITERATIVE_MODES,
    CLASSIFIER_NAMES,
    BOOTSTRAP_ITERATIONS,
    CORR_THS,
    REMOVE_ISOLATED,
):
    # 0. Setup
    test_filters, control_filters = sample_cluster_contrast

    # 1. Multi-level samples annotation
    # 1.1. Annotation of test samples
    goi_level = [key for key in test_filters.keys() if GOI_LEVEL_PREFIX in key]
    contrast_level_test = "_".join(chain(*test_filters.values())) + (
        goi_level[0].replace(GOI_LEVEL_PREFIX, "") if len(goi_level) > 0 else ""
    )

    # 1.2. Annotation of control samples
    goi_level = [key for key in control_filters.keys() if GOI_LEVEL_PREFIX in key]
    contrast_level_control = "_".join(chain(*control_filters.values())) + (
        goi_level[0].replace(GOI_LEVEL_PREFIX, "") if len(goi_level) > 0 else ""
    )

    # 1.3. Set experiment prefix
    contrasts_levels = (contrast_level_test, contrast_level_control)
    exp_prefix = (
        f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
        f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
    )

    # 1.4. Set experiment name
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    exp_name = (
        f"{exp_prefix}_{contrast_level_test}_vs_{contrast_level_control}_"
        + f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
    )

    input_collection.append(
        dict(
            data_root=DATA_ROOT,
            exp_name=exp_name,
            network_type=network_type,
            correlation_type=correlation_type,
            classifier_name=classifier_name,
            bootstrap_iterations=bootstrap_iterations,
            iterative=iterative,
            corr_th=corr_th,
            remove_isolated=remove_isolated,
        )
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=rich_wgcna_network),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            rich_wgcna_network(**ins)
