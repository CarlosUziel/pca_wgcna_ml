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
DATA_ROOT: Path = STORAGE.joinpath("TCGA_PRAD_SU2C_PCF_GSE221601")
ML_ROOT: Path = DATA_ROOT.joinpath("ml_classifiers")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"

CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("HSPC", "PRIM"),
    ("MCRPC", "HSPC"),
]
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
SPECIES: str = "Homo sapiens"
CLASSIFIER_NAMES: Iterable[str] = ("decision_tree", "random_forest")
BOOTSTRAP_ITERATIONS: int = 10000
SHAP_THS: Iterable[float] = (1e-4, 1e-5)
PARALLEL: bool = True


contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
org_db = OrgDB(SPECIES)

# 1. Generate input collection for all arguments' combinations
input_collection = []
for (test, control), p_col, p_th, lfc_level, lfc_th, classifier_name in product(
    CONTRASTS_LEVELS, P_COLS, P_THS, LFC_LEVELS, LFC_THS, CLASSIFIER_NAMES
):
    # 1.1. Setup
    p_thr_str = str(p_th).replace(".", "_")
    lfc_thr_str = str(lfc_th).replace(".", "_")
    data_root = (
        ML_ROOT.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_"
            f"{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}"
        )
        .joinpath(classifier_name)
        .joinpath("genes_features")
    )
    func_path = data_root.joinpath("functional")
    func_path.mkdir(exist_ok=True, parents=True)
    plots_path = data_root.joinpath("plots")
    plots_path.mkdir(exist_ok=True, parents=True)

    exp_name = f"bootstrap_{BOOTSTRAP_ITERATIONS}"
    results_file = data_root.joinpath("bootstrap").joinpath(
        f"{exp_name}_shap_values.csv"
    )

    if not results_file.exists():
        continue

    # 1.2. Add GSEA inputs
    input_collection.append(
        dict(
            data_type="diff_expr_ml",
            func_path=func_path,
            plots_path=plots_path,
            results_file=results_file,
            exp_name=exp_name,
            org_db=org_db,
            analysis_type="gsea",
        )
    )

    # 1.3. Add ORA inputs
    for shap_th in SHAP_THS:
        shap_th_str = str(shap_th).replace(".", "_")
        input_collection.append(
            dict(
                data_type="diff_expr_ml",
                func_path=func_path,
                plots_path=plots_path,
                results_file=results_file,
                exp_name=f"{exp_name}_shap_values_{shap_th_str}",
                org_db=org_db,
                shap_th=shap_th,
                analysis_type="ora",
                cspa_surfaceome_file=STORAGE.joinpath(
                    "CSPA_validated_surfaceome_proteins_human.csv"
                ),
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
