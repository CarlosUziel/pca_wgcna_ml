import argparse
import functools
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.wgcna.utils import differential_methylation
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
DATA_ROOT: Path = STORAGE.joinpath("TCGA_PRAD_MethArray")
WGCNA_ROOT = DATA_ROOT.joinpath("wgcna")
WGCNA_ROOT.mkdir(exist_ok=True, parents=True)
ANNOT_FILE: Path = DATA_ROOT.joinpath("data").joinpath("samples_annotation_common.csv")
GENOME: str = "hg38"
SAMPLE_CONTRAST_FACTOR: Iterable[str] = "sample_type"

CONTRASTS_LEVELS: List[Tuple[str, str]] = [
    ("prim", "norm"),
]
NORM_TYPES: Iterable[str] = ("noob_quantile",)
GENE_ANNOTS: Iterable[str] = (f"{GENOME}_genes_promoters",)
P_COLS: Iterable[str] = ("P.Value", "adj.P.Val")
P_THS: Iterable[float] = (0.05, 0.01)
LFC_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
LFC_THS: Iterable[float] = (0.0, 1.0, 2.0)
MEAN_METH_DIFF_THS: Iterable[float] = (0.1, 0.2, 0.3)
SPECIES: str = "Homo sapiens"
NETWORK_TYPES: Iterable[str] = ("signed",)
CORRELATION_TYPES: Iterable[str] = ("bicor",)
ITERATIVE_MODES: Iterable[bool] = (True, False)
WGCNA_THREADS: int = 4
PARALLEL: bool = True

annot_df = (
    pd.read_csv(ANNOT_FILE, index_col=0, dtype=str)
    .rename(columns={"barcode": "Basename"})
    .sort_values("probe")
    .drop_duplicates("Basename")
)

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
org_db = OrgDB(SPECIES)

input_collection = []
for norm_type, gene_annot, (test, control) in product(
    NORM_TYPES, GENE_ANNOTS, CONTRASTS_LEVELS
):
    exp_prefix = (
        f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
        f"diff_meth_probes_{norm_type}_top_table_{test}_vs_{control}"
    )

    for (
        p_col,
        p_th,
        lfc_level,
        lfc_th,
        mean_meth_diff_th,
        network_type,
        correlation_type,
        iterative,
    ) in product(
        P_COLS,
        P_THS,
        LFC_LEVELS,
        LFC_THS,
        MEAN_METH_DIFF_THS,
        NETWORK_TYPES,
        CORRELATION_TYPES,
        ITERATIVE_MODES,
    ):
        p_col_str = p_col.replace(".", "_")
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_th).replace(".", "_")
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        exp_name = (
            f"{exp_prefix}_sig_{p_col_str}_{p_thr_str}_{lfc_level}_{lfc_thr_str}_"
            f"wrt_mean_diff_{mean_meth_diff_th_str}_{gene_annot}"
        )

        custom_meth_probes_file = DATA_ROOT.joinpath("minfi").joinpath(
            f"{exp_name}.csv"
        )
        if not custom_meth_probes_file.exists():
            continue

        input_collection.append(
            dict(
                meth_values_file=(
                    DATA_ROOT.joinpath("minfi").joinpath(
                        f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
                        f"diff_meth_probes_{norm_type}_{test}_vs_{control}_"
                        f"b_values_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
                    )
                ),
                wgcna_path=(
                    WGCNA_ROOT.joinpath(exp_name).joinpath(
                        "iterative" if iterative else "standard"
                    )
                ),
                annot_df=deepcopy(annot_df),
                custom_meth_probes_file=custom_meth_probes_file,
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
            functools.partial(run_func_dict, func=differential_methylation),
            input_collection,
            threads=60 // WGCNA_THREADS,
        )
    else:
        for ins in tqdm(input_collection):
            differential_methylation(**ins)
