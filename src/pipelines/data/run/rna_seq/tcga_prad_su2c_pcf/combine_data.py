import argparse
import logging
import multiprocessing
import warnings
from pathlib import Path
from typing import Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.data.utils import tcga_prad_su2c_pcf_rna_seq

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
DESEQ2_PATH: Path = STORAGE.joinpath("TCGA_PRAD_SU2C_RNASeq").joinpath("deseq2")
DESEQ2_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = DESEQ2_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
TCGA_PRAD_ANNOT_FILE: Path = DATA_PATH.joinpath("samples_annotation_tcga_prad.csv")
SU2C_PCF_ANNOT_FILE: Path = (
    STORAGE.joinpath("SU2C_PCF_2019_RNASeq")
    .joinpath("samples_annotations")
    .joinpath("samples_annotation_rna_downloaded_standarized.csv")
)
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

tcga_prad_su2c_pcf_rna_seq(
    deseq2_path=DESEQ2_PATH,
    plots_path=PLOTS_PATH,
    data_path=DATA_PATH,
    tcga_prad_annot_file=TCGA_PRAD_ANNOT_FILE,
    su2c_pcf_annot_file=SU2C_PCF_ANNOT_FILE,
    sample_colors=SAMPLE_COLORS,
    sample_type_col=SAMPLE_TYPE_COL,
)
