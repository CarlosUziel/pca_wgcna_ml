import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.functional_analysis.utils import functional_enrichment
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
DATA_ROOT: Path = STORAGE.joinpath("PCTA_WCDT_GSE221601_LRT")
FUNC_PATH: Path = DATA_ROOT.joinpath("functional")
FUNC_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = FUNC_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DESEQ2_ROOT = DATA_ROOT.joinpath("deseq2")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"

P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
SPECIES: str = "Homo sapiens"
PARALLEL: bool = True


exp_prefix = (
    "res_LRT_across_sample_types_overall_effects_hspc+mcrpc+norm+prim_1232samples"
)
org_db = OrgDB(SPECIES)

# 1. Generate input collection for all arguments' combinations
input_collection = []
results_file = DESEQ2_ROOT.joinpath(f"{exp_prefix}_deseq_results_unique.csv")

# 1.1. Add GSEA inputs
if results_file.exists():
    input_collection.append(
        dict(
            data_type="diff_expr",
            func_path=FUNC_PATH,
            plots_path=PLOTS_PATH,
            results_file=results_file,
            exp_name=exp_prefix,
            org_db=org_db,
            analysis_type="gsea",
        )
    )

# 1.2. Add ORA inputs
for p_col, p_th, lfc_level, lfc_th in product(P_COLS, P_THS, LFC_LEVELS, LFC_THS):
    p_thr_str = str(p_th).replace(".", "_")
    lfc_thr_str = str(lfc_th).replace(".", "_")
    input_collection.append(
        dict(
            data_type="diff_expr",
            func_path=FUNC_PATH,
            plots_path=PLOTS_PATH,
            results_file=results_file,
            exp_name=f"{exp_prefix}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}",
            org_db=org_db,
            cspa_surfaceome_file=STORAGE.joinpath(
                "CSPA_validated_surfaceome_proteins_human.csv"
            ),
            p_col=p_col,
            p_th=p_th,
            lfc_col="log2FoldChange",
            lfc_level=lfc_level,
            lfc_th=lfc_th,
            numeric_col="log2FoldChange",
            analysis_type="ora",
        )
    )

# 2. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=functional_enrichment),
            input_collection,
            threads=user_args["threads"] // 3,
        )
    else:
        for ins in tqdm(input_collection):
            functional_enrichment(**ins)
