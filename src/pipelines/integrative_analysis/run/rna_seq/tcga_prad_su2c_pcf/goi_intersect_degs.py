import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.integrative_analysis.utils import intersect_degs
from r_wrappers.utils import map_gene_id
from utils import run_func_dict

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
SAMPLE_CONTRAST_FACTOR: str = "sample_cluster_no_replicates"
GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
CONTRAST_COMPARISONS: Dict[
    str,
    Iterable[
        Tuple[Dict[Iterable[str], Iterable[str]], Dict[Iterable[str], Iterable[str]]]
    ],
] = {
    "prim_norm": (
        (
            {SAMPLE_CONTRAST_FACTOR: ["prim"]},
            {SAMPLE_CONTRAST_FACTOR: ["norm"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["prim"], f"{GOI_LEVEL_PREFIX}_10": ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["norm"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["prim"], f"{GOI_LEVEL_PREFIX}_10": ["low"]},
            {SAMPLE_CONTRAST_FACTOR: ["norm"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["prim"], f"{GOI_LEVEL_PREFIX}_10": ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["prim"], f"{GOI_LEVEL_PREFIX}_10": ["low"]},
        ),
    ),
    "met_prim": (
        (
            {SAMPLE_CONTRAST_FACTOR: ["met"]},
            {SAMPLE_CONTRAST_FACTOR: ["prim"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["prim"], f"{GOI_LEVEL_PREFIX}_10": ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["prim"], f"{GOI_LEVEL_PREFIX}_10": ["low"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["met"], f"{GOI_LEVEL_PREFIX}_10": ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["met"], f"{GOI_LEVEL_PREFIX}_10": ["low"]},
        ),
    ),
}
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

input_collection = []
for contrast_comparison, contrast_comparison_filters in CONTRAST_COMPARISONS.items():
    contrast_prefixes = {}
    # 1. Get contrast file prefixes
    for contrast_filters in contrast_comparison_filters:
        # 1.1. Setup
        test_filters, control_filters = contrast_filters

        # 1.2. Multi-level samples annotation
        # 1.2.1. Annotation of test samples
        goi_level = [key for key in test_filters.keys() if GOI_LEVEL_PREFIX in key]
        contrast_level_test = "_".join(chain(*test_filters.values())) + (
            goi_level[0].replace(GOI_LEVEL_PREFIX, "") if len(goi_level) > 0 else ""
        )

        # 1.2.2. Annotation of control samples
        goi_level = [key for key in control_filters.keys() if GOI_LEVEL_PREFIX in key]
        contrast_level_control = "_".join(chain(*control_filters.values())) + (
            goi_level[0].replace(GOI_LEVEL_PREFIX, "") if len(goi_level) > 0 else ""
        )

        # 1.3. Set experiment prefix and remove unnecesary samples
        contrasts_levels = (contrast_level_test, contrast_level_control)
        exp_prefix = (
            f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
        )

        contrast_prefixes[f"{contrast_level_test}_vs_{contrast_level_control}"] = (
            f"{exp_prefix}_{contrast_level_test}_vs_{contrast_level_control}"
        )

    for p_col, p_th, lfc_level, lfc_th in product(P_COLS, P_THS, LFC_LEVELS, LFC_THS):
        # 2. Generate input collection for all arguments' combinations
        input_collection.append(
            dict(
                contrast_prefixes=contrast_prefixes,
                root_path=DATA_ROOT,
                comparison_alias=contrast_comparison,
                p_col=p_col,
                p_th=p_th,
                lfc_level=lfc_level,
                lfc_th=lfc_th,
            )
        )

# 3. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=intersect_degs),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            intersect_degs(**ins)
