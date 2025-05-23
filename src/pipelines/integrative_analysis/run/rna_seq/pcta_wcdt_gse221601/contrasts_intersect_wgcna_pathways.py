import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Tuple

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from data.utils import parallelize_map
from pipelines.integrative_analysis.utils import intersect_wgcna_pathways
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
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
    ("hspc", "prim"),
    ("mcrpc", "hspc"),
]
CONTRAST_COMPARISONS: Dict[str, Iterable[Iterable[str]]] = {
    "comparison_0": (
        ("prim", "norm"),
        ("hspc", "prim"),
        ("mcrpc", "hspc"),
    ),
}
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
NETWORK_TYPES: Iterable[str] = ("signed",)
CORRELATION_TYPES: Iterable[str] = ("bicor",)
FUNC_DBS: Iterable[str] = (
    "KEGG",
    "REACTOME",
    "DO",
    "NCG",
    "MKEGG",
    "GO_ALL",
    "GO_BP",
    "GO_CC",
    "GO_MF",
)
PARALLEL: bool = True

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"

input_collection = []
for contrast_comparison, contrast_comparison_filters in CONTRAST_COMPARISONS.items():
    contrast_prefixes = {
        f"{test}_vs_{control}": f"{exp_prefix}_{test}_vs_{control}"
        for test, control in contrast_comparison_filters
    }

    for (
        p_col,
        p_th,
        lfc_level,
        lfc_th,
        correlation_type,
        network_type,
        func_db,
    ) in product(
        P_COLS, P_THS, LFC_LEVELS, LFC_THS, CORRELATION_TYPES, NETWORK_TYPES, FUNC_DBS
    ):
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
                correlation_type=correlation_type,
                network_type=network_type,
                func_db=func_db,
            )
        )

# 3. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=intersect_wgcna_pathways),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            intersect_wgcna_pathways(**ins)
