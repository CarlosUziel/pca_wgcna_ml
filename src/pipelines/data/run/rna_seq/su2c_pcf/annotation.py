import argparse
import logging
import multiprocessing
import warnings
from pathlib import Path

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.data.utils import su2c_pcf_annotation

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
DATA_ROOT: Path = STORAGE.joinpath("SU2C_PCF_2019_RNASeq")
DATA_PATH: Path = DATA_ROOT.joinpath("samples_annotations")
RNA_FASTQ_PATH: Path = DATA_ROOT.joinpath("fastq_raw")
DNA_FASTQ_PATH: Path = STORAGE.joinpath("SU2C_PCF_2019_WES").joinpath("fastq_raw")

su2c_pcf_annotation(
    data_path=DATA_PATH, rna_fastq_path=RNA_FASTQ_PATH, dna_fastq_path=DNA_FASTQ_PATH
)
