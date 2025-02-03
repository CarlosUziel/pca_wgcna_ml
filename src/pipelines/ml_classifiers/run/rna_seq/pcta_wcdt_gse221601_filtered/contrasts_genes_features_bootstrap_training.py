import argparse
import functools
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain, product
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
DATA_ROOT: Path = STORAGE.joinpath("PCTA_WCDT_GSE221601_FILTERED")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")
DESEQ2_ROOT = DATA_ROOT.joinpath("deseq2")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
    ("hspc", "prim"),
    ("mcrpc", "hspc"),
]
SPECIES: str = "Homo sapiens"
CLASSIFIER_NAMES: Iterable[str] = (
    "decision_tree",
    "random_forest",
    "light_gbm",
    "mlp",
    "tabpfn",
)
BOOTSTRAP_ITERATIONS: int = 10000
PARALLEL: bool = False

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
results_prefix = (
    "comparison_0_padj_0_05_up_1_0_bicor_signed_intersecting_wgcna_M3+M2+M1_ENTREZID_22"
)
org_db = OrgDB(SPECIES)
annot_df = pd.read_csv(ANNOT_PATH, index_col=0)

wgcna_ml_results = pd.read_csv(
    DATA_ROOT
    / "integrative_analysis"
    / "intersecting_wgcna"
    / Path(results_prefix).with_suffix(".csv"),
    index_col=0,
)
wgcna_ml_results_filt = wgcna_ml_results.loc[
    wgcna_ml_results["prim_vs_norm_M3"]
    & wgcna_ml_results["hspc_vs_prim_M2"]
    & wgcna_ml_results["mcrpc_vs_hspc_M1"]
]

input_collection = []
for (test, control), classifier_name in product(CONTRASTS_LEVELS, CLASSIFIER_NAMES):
    exp_name = f"{exp_prefix}_{test}_vs_{control}_{results_prefix}"

    annot_df_contrasts = deepcopy(
        annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin((test, control))]
    )

    input_collection.append(
        dict(
            data=(DESEQ2_ROOT.joinpath(f"{exp_prefix}_vst.csv")),
            annot_df=annot_df_contrasts,
            contrast_factor=SAMPLE_CONTRAST_FACTOR,
            org_db=org_db,
            classifier_name=classifier_name,
            hparams_file=(
                DATA_ROOT.joinpath("ml_classifiers")
                .joinpath(results_prefix)
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("tuning")
                .joinpath("best_hparams.json")
            ),
            results_path=(
                DATA_ROOT.joinpath("ml_classifiers")
                .joinpath(results_prefix)
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("bootstrap")
            ),
            custom_features=wgcna_ml_results_filt,
            custom_features_gene_type="ENTREZID",
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
