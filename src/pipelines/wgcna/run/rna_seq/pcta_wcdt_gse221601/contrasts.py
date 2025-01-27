import argparse
import functools
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.wgcna.utils import differential_expression
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
ANNOT_PATH: Path = DATA_ROOT.joinpath("data").joinpath("samples_annotation.csv")
WGCNA_ROOT = DATA_ROOT.joinpath("wgcna")
WGCNA_ROOT.mkdir(exist_ok=True, parents=True)
DESEQ2_ROOT = DATA_ROOT.joinpath("deseq2")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
    ("hspc", "prim"),
    ("mcrpc", "hspc"),
    ("mcrpc", "prim"),
]
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
SPECIES: str = "Homo sapiens"
NETWORK_TYPES: Iterable[str] = ("signed",)
CORRELATION_TYPES: Iterable[str] = ("bicor",)
ITERATIVE_MODES: Iterable[bool] = (True, False)
WGCNA_THREADS: int = 4
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
org_db = OrgDB(SPECIES)

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
) in product(
    CONTRASTS_LEVELS,
    P_COLS,
    P_THS,
    LFC_LEVELS,
    LFC_THS,
    NETWORK_TYPES,
    CORRELATION_TYPES,
    ITERATIVE_MODES,
):
    p_thr_str = str(p_th).replace(".", "_")
    lfc_thr_str = str(lfc_th).replace(".", "_")
    exp_name = (
        f"{exp_prefix}_{test}_vs_{control}_"
        + f"{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}"
    )

    degs_file = DESEQ2_ROOT.joinpath(f"{exp_name}_deseq_results_unique.csv")
    if not degs_file.exists():
        continue

    input_collection.append(
        dict(
            data_file=DESEQ2_ROOT.joinpath(f"{exp_prefix}_vst.csv"),
            wgcna_path=(
                WGCNA_ROOT.joinpath(exp_name).joinpath(
                    "iterative" if iterative else "standard"
                )
            ),
            degs_file=degs_file,
            annot_df=deepcopy(annot_df),
            contrast_factor=SAMPLE_CONTRAST_FACTOR,
            contrast=(test, control),
            org_db=org_db,
            network_type=network_type,
            correlation_type=correlation_type,
            threads=WGCNA_THREADS,
            iterative=iterative,
        )
    )


if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=differential_expression),
            input_collection,
            threads=60 // WGCNA_THREADS,
        )
    else:
        for ins in tqdm(input_collection):
            differential_expression(**ins)
