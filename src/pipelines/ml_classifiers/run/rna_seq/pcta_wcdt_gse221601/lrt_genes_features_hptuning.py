import argparse
import functools
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain, product
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.ml_classifiers.utils import hparams_tuning
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
DATA_ROOT: Path = STORAGE.joinpath("PCTA_WCDT_GSE221601")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("HSPC", "PRIM"),
    ("MCRPC", "HSPC"),
]
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "PRIM": "#4A708B",
    "MCRPC": "#8B3A3A",
    "HSPC": "purple",
}
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
SPECIES: str = "Homo sapiens"
CLASSIFIER_NAMES: Iterable[str] = ("decision_tree", "random_forest")
PARALLEL: bool = True

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = (
    "Sig_res_LRT_across_sample_types_overall_effects_hspc+mcrpc+norm+prim_1232samples"
)
org_db = OrgDB(SPECIES)
annot_df = pd.read_csv(ANNOT_PATH, index_col=0)

input_collection = []
for p_col, p_th, lfc_level, lfc_th, classifier_name in product(
    P_COLS, P_THS, LFC_LEVELS, LFC_THS, CLASSIFIER_NAMES
):
    p_thr_str = str(p_th).replace(".", "_")
    lfc_thr_str = str(lfc_th).replace(".", "_")
    exp_name = f"{exp_prefix}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}"

    custom_genes_file = DATA_ROOT.joinpath("deseq2_lrt").joinpath(
        f"{exp_name}_deseq_results_unique.csv"
    )

    if not custom_genes_file.exists():
        continue

    annot_df_contrasts = deepcopy(
        annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin(contrast_conditions)]
    )

    input_collection.append(
        dict(
            data_type="gene_expr",
            features_type="genes",
            classifier_name=classifier_name,
            data=DATA_ROOT.joinpath("deseq2_lrt").joinpath(
                "vsd_filtered_LRT_reduced_design_sample_types_hspc+mcrpc+norm+prim.csv"
            ),
            annot_df=annot_df_contrasts,
            contrast_factor=SAMPLE_CONTRAST_FACTOR,
            hparams_grid_file=(
                Path(__file__)
                .resolve()
                .parents[3]
                .joinpath("hparams_grids")
                .joinpath(f"{classifier_name}.json")
            ),
            org_db=org_db,
            contrasts_levels_colors=CONTRASTS_LEVELS_COLORS,
            results_path=(
                DATA_ROOT.joinpath("ml_classifiers_lrt")
                .joinpath(exp_name)
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("tuning")
            ),
            custom_features_file=custom_genes_file,
            random_seed=8080,
        )
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=hparams_tuning),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            hparams_tuning(**ins)
