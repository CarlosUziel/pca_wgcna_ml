import argparse
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from components.functional_analysis.orgdb import OrgDB
from pipelines.differential_expression.utils import differential_expression

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
DATA_ROOT: Path = STORAGE.joinpath("PCTA_WCDT_GSE221601")
RESULTS_PATH: Path = DATA_ROOT.joinpath("deseq2")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = RESULTS_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")
COUNTS_MATRIX_PATH: Path = DATA_PATH.joinpath("all_counts_batch_corrected_v3.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
    ("hspc", "prim"),
    ("mcrpc", "hspc"),
    ("mcrpc", "prim"),
]
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "norm": "#1A6566",
    "prim": "#21445B",
    "hspc": "#323050",
    "mcrpc": "#45214A",
}
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
HEATMAP_TOP_N: int = 1000
COMPUTE_VST: bool = True
COMPUTE_RLOG: bool = False
SPECIES: str = "Homo sapiens"

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
org_db = OrgDB(SPECIES)

counts_df = pd.read_csv(COUNTS_MATRIX_PATH, index_col=0)

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"

annot_df_contrasts = deepcopy(
    annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin(contrast_conditions)]
)

# 1. Generate input collection for all arguments' combinations
input_collection = dict(
    annot_df_contrasts=annot_df_contrasts,
    results_path=RESULTS_PATH,
    plots_path=PLOTS_PATH,
    exp_prefix=exp_prefix,
    org_db=org_db,
    factors=[SAMPLE_CONTRAST_FACTOR],
    contrast_factor=SAMPLE_CONTRAST_FACTOR,
    contrasts_levels=CONTRASTS_LEVELS,
    contrast_levels_colors=CONTRASTS_LEVELS_COLORS,
    p_cols=P_COLS,
    p_ths=P_THS,
    lfc_levels=LFC_LEVELS,
    lfc_ths=LFC_THS,
    heatmap_top_n=HEATMAP_TOP_N,
    counts_matrix=counts_df,
    compute_vst=COMPUTE_VST,
    compute_rlog=COMPUTE_RLOG,
)


# 2. Run differential expression analysis
if __name__ == "__main__":
    differential_expression(**input_collection)
