import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_star

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

ROOT: Path = Path("/gpfs/data/fs71358/cperez/storage")
GENOMES_PATH: Path = ROOT.joinpath("genomes/Homo_sapiens/GRCh38/ENSEMBL")
STORAGE: Path = ROOT.joinpath("SU2C_PCF_2019_RNASeq")
FASTQ_PATH: Path = STORAGE.joinpath("cutadapt")
STAR_PATH: Path = STORAGE.joinpath("mapping/star")
INDEX_PATH: Path = GENOMES_PATH.joinpath("star_index")
# https://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/
# Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
GENOME_FASTA_FILES: Iterable[Path] = (
    GENOMES_PATH.joinpath("Homo_sapiens.GRCh38.dna.primary_assembly.fa"),
)
# https://ftp.ensembl.org/pub/release-107/gtf/
# homo_sapiens/Homo_sapiens.GRCh38.107.gtf.gz
GTF_FILE: Path = GENOMES_PATH.joinpath("Homo_sapiens.GRCh38.107.gtf")
RUN_MODE: str = "use_index"  # "use_index" / "create_index"
STAR_KWARGS: Dict[str, Any] = {
    "--runThreadN": 96,
    # below only when RUN_MODE=="use_index"
    "--readFilesCommand": "gunzip -c",
    "--outSAMtype": "BAM SortedByCoordinate",
    "--outBAMcompression": 10,
    "--quantMode": "GeneCounts",
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "STAR",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*.fastq.gz"

run_star(
    fastq_path=FASTQ_PATH,
    star_path=STAR_PATH,
    genome_path=INDEX_PATH,
    star_kwargs=STAR_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    genome_fasta_files=GENOME_FASTA_FILES,
    gtf_file=GTF_FILE,
    run_mode=RUN_MODE,
    pattern=PATTERN,
)
