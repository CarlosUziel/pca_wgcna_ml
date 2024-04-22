"""
This script spawns multiple slurm jobs, each processing one sample. This allows
faster processing of a whole batch of samples, since each sample is run in
parallel (depending on the cluster resources).
"""

import logging
from collections import defaultdict
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tqdm.rich import tqdm

from data.utils import run_cmd
from slurm.slurm_job_submitter import SlurmJobSubmitter


def run_fastqc(
    fastq_path: Path,
    fastqc_path: Path,
    fastqc_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run fastqc on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing fastq files.
        fastqc_path: Path to trimmed reads.
        fastqc_kwargs: A dictionary of FastQC options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run fastqc for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = fastqc_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "fastqc",
                *sample_reads[:2],
                "-o",
                sample_out_path,
                *list(chain(*fastqc_kwargs.items())),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.fastqc.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.fastqc.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_cutadapt(
    fastq_path: Path,
    cutadapt_path: Path,
    fwd_adapter_file: Path,
    rv_adapter_file: Path,
    cutadapt_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run cutadapt on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing fastq files.
        cutadapt_path: Path to store trimmed reads.
        fwd_adapter_file: File containing forward adapter sequences.
        rv_adapter_file: File containing reverse adapter sequences.
        cutadapt_kwargs: A dictionary of cutadapt options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run Cutadapt for each sample
    logging.info("Submitting SLURM jobs: ")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = cutadapt_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        read_1_trim = sample_out_path.joinpath(f"{sample_id}.1.trimmed.fastq.gz")
        read_2_trim = sample_out_path.joinpath(f"{sample_id}.2.trimmed.fastq.gz")

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "cutadapt",
                *sample_reads[:2],
                "-b",
                f"file:{fwd_adapter_file}",
                "-o",
                read_1_trim,
                *(
                    ["-B", f"file:{rv_adapter_file}", "-p", read_2_trim]
                    if len(sample_reads) > 1
                    else []
                ),
                *list(chain(*cutadapt_kwargs.items())),
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.cutadapt.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.cutadapt.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_trim_galore(
    fastq_path: Path,
    trim_galore_path: Path,
    trim_galore_kwargs: Dict[str, Any],
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run trim galore on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Args:
        fastq_path: Path to directory containing fastq files.
        cutadapt_path: Path to store trimmed reads.
        cutadapt_kwargs: A dictionary of cutadapt options.
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 2. Run Cutadapt for each sample
    logging.info("Submitting SLURM jobs: ")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 2.1. Create intermediary paths
        sample_out_path = trim_galore_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "trim_galore",
                "--paired",
                *list(chain(*trim_galore_kwargs.items())),
                "-o",
                sample_out_path,
                *sample_reads[:2],
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.trim_galore.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.trim_galore.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_multiqc(
    multiqc_path: Path,
    analyses_paths: Iterable[Path],
    multiqc_kwargs: Dict[str, Any],
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate multiqc reports based on the results paths given.

    Args:
        multiqc_path: Path to store multiqc reports.
        analyses_paths: List of paths to obtain analysis results from.
        multiqc_kwargs: A dictionary of MULTIQC options.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    multiqc_path.mkdir(exist_ok=True, parents=True)

    # 1. Build command
    cmd_args = map(
        str,
        [
            "multiqc",
            *list(chain(*multiqc_kwargs.items())),
            "-o",
            multiqc_path,
            *analyses_paths,
        ],
    )

    # 2. Run command
    if slurm_kwargs is not None:
        slurm_kwargs["--error"] = str(multiqc_path.joinpath("multiqc.error.log"))
        slurm_kwargs["--output"] = str(multiqc_path.joinpath("multiqc.output.log"))

        log_file = multiqc_path.joinpath("multiqc.sbatch.log")
        SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=multiqc_path).submit(
            " ".join(cmd_args), "multi_qc", log_file
        )
    else:
        log_file = multiqc_path.joinpath("multiqc.log")
        run_cmd(cmd=cmd_args, log_path=log_file)


class RunMode(str, Enum):
    create_index = "create_index"
    use_index = "use_index"


def run_star(
    fastq_path: Path,
    star_path: Path,
    genome_path: Path,
    star_kwargs: Dict[str, Any],
    genome_fasta_files: Iterable[Path] = None,
    gtf_file: Path = None,
    run_mode: RunMode = RunMode.use_index,
    pattern: str = "**/*.fastq.gz",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run star on fastq files. Expected naming scheme of fastq files is:
        {sample_id}.{pair_n}.fastq.gz

    Genome files can be obtained from the ENSEMBL ftp server. For example, Homo
    sapiens files can be obtained from:
        https://www.ensembl.org/Homo_sapiens/Info/Index

    Args:
        fastq_path: Path to directory containing fastq files.
        star_path: Path to store aligned reads.
        genome_path: Location of genome files, generated by star in a previous
            step.
        star_kwargs: A list of star options. Option keys followed by
            their values (e.g. [--an-option, value]).
        genome_fasta_files: Paths to the fasta files with the genome sequences.
        gtf_file: path to the GTF file with annotations.
        run_mode: Whether to run star for alignment of reads or to create a
            genome index (necessary for alignment).
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Setup
    genome_path.mkdir(exist_ok=True, parents=True)
    assert len(list(genome_path.glob("*"))) != 0 or run_mode == RunMode.create_index, (
        f"{genome_path} is empty. Either select a valid path containing "
        "genome indices or generate them by running the command again "
        'using the "--run-mode create_index"'
    )

    # 2. Create index if requested
    if run_mode == RunMode.create_index:
        assert gtf_file is not None and genome_fasta_files is not None, (
            "gtf_file and genome_fasta_files have to be provided when "
            "creating a genome index."
        )

        # 2.1. Build command
        cmd_args = map(
            str,
            [
                "STAR",
                "--runMode",
                "genomeGenerate",
                "--genomeDir",
                genome_path,
                "--genomeFastaFiles",
                *genome_fasta_files,
                "--sjdbGTFfile",
                gtf_file,
                *list(chain(*star_kwargs.items())),
            ],
        )

        # 2.2. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(genome_path.joinpath("star_genome.error.log"))
            slurm_kwargs["--output"] = str(
                genome_path.joinpath("star_genome.star.output.log")
            )

            log_file = genome_path.joinpath("create_index.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=genome_path).submit(
                " ".join(cmd_args), "create_index", log_file
            )
            logging.info("Submitted SLURM job to create genome index.")
        else:
            log_file = genome_path.joinpath("create_index.log")
            run_cmd(cmd=cmd_args, log_path=log_file)

        return

    # 3. Get all fastq files per sample
    samples_reads = defaultdict(list)
    for reads_file in sorted(fastq_path.glob(pattern)):
        sample_id = reads_file.stem.partition(".")[0]
        samples_reads[sample_id].append(reads_file)

    # 4. Run star for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, sample_reads in tqdm(samples_reads.items()):
        # 4.1. Create intermediary paths
        sample_out_path = star_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)

        # 4.2. Build command
        cmd_args = map(
            str,
            [
                "STAR",
                "--runMode",
                "alignReads",
                "--genomeDir",
                genome_path,
                "--readFilesIn",
                *sample_reads,
                "--outFileNamePrefix",
                str(sample_out_path) + "/",
                *list(chain(*star_kwargs.items())),
            ],
        )

        # 4.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.star.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.star.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)


def run_htseq_count(
    bam_path: Path,
    htseq_path: Path,
    gtf_file: Path,
    htseq_kwargs: Dict[str, Any],
    pattern: str = "**/*.bam",
    slurm_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run htseq-count on bam files.

    Args:
        bam_path: Path to directory containing fastq files.
        htseq_path: Path to store htseq files.
        gtf_file: path to the GTF file with annotations.
        htseq_kwargs: A list of htseq-counts options. Option keys followed by
            their values (e.g. [--an-option, value]).
        pattern: File name pattern the fastq files to be processed should follow.
        slurm_kwargs: A dictionary of SLURM cluster batch job options. If missing, each
            command will be run locally in sequential order.
    """
    # 1. Get all bam files
    samples_bams = {}
    for bam_file in sorted(bam_path.glob(pattern)):
        sample_id = bam_file.parent.name
        samples_bams[sample_id] = bam_file

    # 2. Run fastqc for each sample
    logging.info("Submitting SLURM jobs:")
    for sample_id, bam_file in tqdm(samples_bams.items()):
        # 2.1. Create intermediary paths
        sample_out_path = htseq_path.joinpath(sample_id)
        sample_out_path.mkdir(parents=True, exist_ok=True)
        htseq_out = sample_out_path.joinpath(f"{sample_id}.tsv")

        # 2.2. Build command
        cmd_args = map(
            str,
            [
                "htseq-count",
                *list(chain(*htseq_kwargs.items())),
                "-c",
                htseq_out,
                bam_file,
                gtf_file,
            ],
        )

        # 2.3. Run command
        if slurm_kwargs is not None:
            slurm_kwargs["--error"] = str(
                sample_out_path.joinpath(f"{sample_id}.htseq_count.error.log")
            )
            slurm_kwargs["--output"] = str(
                sample_out_path.joinpath(f"{sample_id}.htseq_count.output.log")
            )

            log_file = sample_out_path.joinpath(f"{sample_id}.sbatch.log")
            SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=sample_out_path).submit(
                " ".join(cmd_args), sample_id, log_file
            )
        else:
            log_file = sample_out_path.joinpath(f"{sample_id}.log")
            run_cmd(cmd=cmd_args, log_path=log_file)
