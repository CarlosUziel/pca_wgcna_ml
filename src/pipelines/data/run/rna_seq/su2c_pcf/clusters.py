import argparse
import logging
import multiprocessing
import warnings
from pathlib import Path
from typing import Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.data.utils import su2c_pcf_clusters

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
ANNOT_FILE: Path = DATA_PATH.joinpath("samples_annotation_tcga_prad_su2c.csv")
SAMPLE_COLORS: Dict[str, str] = {
    "norm": "#9ACD32",
    "prim": "#4A708B",
    "met": "red",
    "met_b": "darkred",
    "met_bb": "#8B3A3A",
    "met_a": "orange",
    "met_aa": "darkorange",
    "endo": "purple",
}
SAMPLE_TYPE_COL: str = "sample_type"
SAMPLE_CLUSTER_FIELD: str = "sample_cluster_no_replicates"

su2c_pcf_clusters(
    root_path=DATA_ROOT,
    annot_file=ANNOT_FILE,
    sample_colors=SAMPLE_COLORS,
    sample_type_col=SAMPLE_TYPE_COL,
    sample_cluster_field=SAMPLE_CLUSTER_FIELD,
)
