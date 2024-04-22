import argparse
import logging
import multiprocessing
import warnings
from itertools import chain, product
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

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
WGCNA_ROOT = DATA_ROOT.joinpath("wgcna")
ML_ROOT = DATA_ROOT.joinpath("ml_classifiers")
SAMPLE_CONTRAST_FACTOR: str = "sample_cluster_no_replicates"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
    ("met_bb", "prim"),
]
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
NETWORK_TYPES: Iterable[str] = ("signed",)
CORRELATION_TYPES: Iterable[str] = ("bicor",)
ITERATIVE_MODES: Iterable[bool] = (True, False)
CLASSIFIER_NAMES: Iterable[str] = ("decision_tree", "random_forest", "light_gbm")
BOOTSTRAP_ITERATIONS: int = 10000
SHAP_THS: Iterable[float] = (1e-3, 1e-4, 1e-5)


contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"

# 1. Generate input collection for all arguments' combinations
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
):
    # 1.1. Setup
    p_thr_str = str(p_th).replace(".", "_")
    lfc_thr_str = str(lfc_th).replace(".", "_")
    wgcna_data_root = WGCNA_ROOT.joinpath(
        f"{exp_prefix}_{test}_vs_{control}_"
        f"{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}"
    ).joinpath("iterative" if iterative else "standard")
    ml_data_root = ML_ROOT.joinpath(
        f"{exp_prefix}_{test}_vs_{control}_"
        f"{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}"
    ).joinpath(classifier_name)

    wgcna_exp_name = f"{correlation_type}_{network_type}"
    ml_exp_name = f"bootstrap_{BOOTSTRAP_ITERATIONS}"

    ml_results = pd.read_csv(
        ml_data_root.joinpath("bootstrap").joinpath(f"{ml_exp_name}_shap_values.csv"),
        index_col=0,
    )

    # 2. Run filtering
    module_files = [
        module_file
        for module_file in wgcna_data_root.joinpath("results").glob(
            f"{wgcna_exp_name}_M*_genes.csv"
        )
        if "M0" not in module_file.stem
    ]
    if len(module_files) > 0:
        for module_file in module_files:
            module_name = str(module_file).split("_")[-3]

            module_genes = pd.read_csv(module_file, index_col=0)

            for shap_th in SHAP_THS:
                shap_th_str = str(shap_th).replace(".", "_")
                ml_results_shap = ml_results[abs(ml_results["shap_value"]) > shap_th]
                intersecting_genes = module_genes.index.intersection(
                    ml_results_shap.index
                )

                if len(intersecting_genes) > 0:
                    module_genes.loc[intersecting_genes, :].to_csv(
                        f"{module_file.with_suffix('')}_shap_{shap_th_str}.csv"
                    )
