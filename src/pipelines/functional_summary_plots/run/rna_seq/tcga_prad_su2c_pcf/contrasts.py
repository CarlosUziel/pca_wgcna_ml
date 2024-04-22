import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable, Tuple

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from data.utils import parallelize_map
from pipelines.functional_summary_plots.utils import functional_summary_plots
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
DESEQ_PATH = DATA_ROOT.joinpath("deseq2")
PLOTS_PATH = DATA_ROOT.joinpath("plots")
FUNC_PATH = DATA_ROOT.joinpath("functional")
SAMPLE_CONTRAST_FACTOR: str = "sample_cluster_no_replicates"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
    ("met_bb", "prim"),
]
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
GENE_DATABASES: Iterable[str] = (
    "DO",
    "GO",
    "KEGG",
    "MKEGG",
    "MSIGDB",
    "NCG",
    "REACTOME",
)
PARALLEL: bool = True

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"

# 1. Generate input collection for all arguments' combinations
input_collection = []
for (test, control), gene_database in product(CONTRASTS_LEVELS, GENE_DATABASES):
    # 1.1. Setup
    exp_name = f"{exp_prefix}_{test}_vs_{control}"

    # 1.2. Add GSEA inputs
    functional_result_file = FUNC_PATH.joinpath(gene_database).joinpath(
        f"{exp_name}_gsea.csv"
    )
    if functional_result_file.exists():
        input_collection.append(
            dict(
                functional_result_file=functional_result_file,
                save_path_prefix=PLOTS_PATH.joinpath(gene_database),
            )
        )

    # 1.3. Add ORA inputs
    for (
        p_col,
        p_th,
        lfc_level,
        lfc_th,
    ) in product(P_COLS, P_THS, LFC_LEVELS, LFC_THS):
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_th).replace(".", "_")

        functional_result_file = FUNC_PATH.joinpath(gene_database).joinpath(
            f"{exp_name}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}_ora.csv"
        )

        if functional_result_file.exists():
            input_collection.append(
                dict(
                    functional_result_file=functional_result_file,
                    save_path_prefix=PLOTS_PATH.joinpath(gene_database),
                )
            )

# 2. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=functional_summary_plots),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            functional_summary_plots(**ins)
