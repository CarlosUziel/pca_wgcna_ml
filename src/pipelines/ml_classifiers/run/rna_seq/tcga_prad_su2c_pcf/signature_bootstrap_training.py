import argparse
import functools
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.ml_classifiers.utils import bootstrap_training
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
DATA_ROOT: Path = STORAGE.joinpath("TCGA_PRAD_SU2C_RNASeq")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation_tcga_prad_su2c_clusters.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_cluster_no_replicates"

CONTRAST_FILES: Iterable[Tuple[Tuple[str, str], Path]] = [
    (
        ("prim", "norm"),
        DATA_ROOT.joinpath("top_signatures_atchclust_k2km2_upreg_privsNR_1435.csv"),
    ),
    (
        ("met_bb", "prim"),
        DATA_ROOT.joinpath("top_signatures_atchclust_k2_km2_upreg_MetvsPri_4760.csv"),
    ),
]
SPECIES: str = "Homo sapiens"
CLASSIFIER_NAMES: Iterable[str] = ("decision_tree", "random_forest", "light_gbm")
BOOTSTRAP_ITERATIONS: int = 10000
PARALLEL: bool = True

org_db = OrgDB(SPECIES)
exp_prefix = "sample_cluster_no_replicates_met_bb+Normal+PRIM"
annot_df = pd.read_csv(ANNOT_PATH, index_col=0)

input_collection = []
for ((test, control), custom_genes_file), classifier_name in product(
    CONTRAST_FILES, CLASSIFIER_NAMES
):
    exp_name = f"{exp_prefix}_{test}_vs_{control}_padj_0_05_up_1_0_signature"

    annot_df_contrasts = deepcopy(
        annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin((test, control))]
    )

    input_collection.append(
        dict(
            data_type="gene_expr",
            features_type="genes",
            classifier_name=classifier_name,
            data=(DATA_ROOT.joinpath("deseq2").joinpath(f"{exp_prefix}_vst.csv")),
            annot_df=annot_df_contrasts,
            contrast_factor=SAMPLE_CONTRAST_FACTOR,
            hparams_file=(
                DATA_ROOT.joinpath("ml_classifiers")
                .joinpath(exp_name)
                .joinpath(classifier_name)
                .joinpath("tuning")
                .joinpath("best_hparams.json")
            ),
            org_db=org_db,
            results_path=(
                DATA_ROOT.joinpath("ml_classifiers")
                .joinpath(exp_name)
                .joinpath(classifier_name)
                .joinpath("bootstrap")
            ),
            custom_features_file=custom_genes_file,
            exclude_features=None,
            bootstrap_iterations=BOOTSTRAP_ITERATIONS,
            random_seed=8080,
        )
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=bootstrap_training),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            bootstrap_training(**ins)
