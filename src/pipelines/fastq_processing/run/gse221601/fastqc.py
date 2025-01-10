import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_fastqc

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/rawdata/GSE221601")
FASTQ_PATH: Path = STORAGE.joinpath("cutadapt")
FASTQC_PATH: Path = STORAGE.joinpath("fastqc_clean")
FASTQC_KWARGS: Dict[str, Any] = {
    "--threads": 64,
}
PATTERN: str = "**/*.fastq.gz"
SLURM_KWARGS: Dict[str, Any] = None

run_fastqc(
    fastq_path=FASTQ_PATH,
    fastqc_path=FASTQC_PATH,
    fastqc_kwargs=FASTQC_KWARGS,
    pattern=PATTERN,
    slurm_kwargs=SLURM_KWARGS,
)
