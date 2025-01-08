import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_multiqc

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/rawdata/GSE221601")
MULTIQC_PATH: Path = STORAGE.joinpath("multiqc_clean")
ANALYSES_PATH: Iterable[Path] = [
    STORAGE.joinpath("fastqc_raw"),
    STORAGE.joinpath("mapping/star"),
]
MULTIQC_KWARGS: Dict[str, Any] = {
    "--interactive": "",
}
SLURM_KWARGS: Dict[str, Any] = None

run_multiqc(
    multiqc_path=MULTIQC_PATH,
    analyses_paths=ANALYSES_PATH,
    multiqc_kwargs=MULTIQC_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
)
