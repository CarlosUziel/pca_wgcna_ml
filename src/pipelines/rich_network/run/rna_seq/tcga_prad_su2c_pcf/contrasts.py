import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import chain, product
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from data.utils import parallelize_map
from pipelines.rich_network.utils import rich_wgcna_network
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
DATA_ROOT: Path = STORAGE.joinpath("TCGA_PRAD_SU2C_RNASeq")
SAMPLE_CONTRAST_FACTOR: str = "sample_cluster_no_replicates"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
    ("met_bb", "prim"),
]
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
SPECIES: str = "Homo sapiens"
NETWORK_TYPES: Iterable[str] = ("signed",)
CORRELATION_TYPES: Iterable[str] = ("bicor",)
ITERATIVE_MODES: Iterable[bool] = (True, False)
CLASSIFIER_NAMES: Iterable[str] = ("decision_tree", "random_forest", "light_gbm")
BOOTSTRAP_ITERATIONS: Iterable[int] = (10000,)
CORR_THS: Iterable[float] = trunc(np.arange(0.1, 1, 0.1), decs=1)
REMOVE_ISOLATED: Iterable[bool] = (True, False)
PARALLEL: bool = True


contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"

input_collection = []
for (
    (test, control),
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
    CONTRASTS_LEVELS,
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
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    exp_name = (
        f"{exp_prefix}_{test}_vs_{control}_"
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
            threads=32,
        )
    else:
        for ins in tqdm(input_collection):
            rich_wgcna_network(**ins)
