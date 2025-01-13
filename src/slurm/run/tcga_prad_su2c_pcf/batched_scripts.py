"""
Run a series of python scripts in batches, submited as SLURM jobs.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, Union

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from slurm.utils import submit_batches

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


ROOT_PATH: Path = Path("/gpfs/data/fs71358/cperez")
STORAGE_ROOT: Path = ROOT_PATH.joinpath("storage")
SRC_ROOT: Path = ROOT_PATH.joinpath("biopipes").joinpath("src")
COMMON_KWARGS: Dict[str, str] = {"--root-dir": STORAGE_ROOT, "--threads": 96}
SLURM_KWARGS: Dict[str, Union[str, int]] = {
    "--nodes": 1,
    "--partition": "skylake_0384",
    "--qos": "skylake_0384",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
LOGS_PATH: Path = ROOT_PATH.joinpath("logs").joinpath("biopipes")
CONTRASTS_BATCHES: Dict[str, Iterable[Path]] = {
    "contrasts_rna_functional_analysis": (
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_degs.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("functional_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_pathways.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_pathways_genes.py"),
    ),
    "contrasts_rna_ml_genes": (
        SRC_ROOT.joinpath("pipelines")
        .joinpath("ml_classifiers")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_genes_features_hptuning.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("ml_classifiers")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_genes_features_bootstrap_training.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_degs_shap.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_pathways_genes_shap.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_wgcna_shap.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("functional_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_ml.py"),
    ),
    "contrasts_rna_wgcna": (
        SRC_ROOT.joinpath("pipelines")
        .joinpath("wgcna")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_wgcna.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("functional_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_wgcna.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("integrative_analysis")
        .joinpath("run")
        .joinpath("rna_seq")
        .joinpath("tcga_prad_su2c_pcf")
        .joinpath("contrasts_intersect_wgcna_pathways.py"),
    ),
    "contrasts_meth": (
        SRC_ROOT.joinpath("pipelines")
        .joinpath("differential_methylation")
        .joinpath("run")
        .joinpath("tcga_prad")
        .joinpath("contrasts.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("functional_analysis")
        .joinpath("run")
        .joinpath("meth_array")
        .joinpath("tcga_prad")
        .joinpath("contrasts.py"),
    ),
}


submit_batches(
    batches={**CONTRASTS_BATCHES},
    src_path=SRC_ROOT,
    logs_path=LOGS_PATH,
    common_kwargs=COMMON_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
)
