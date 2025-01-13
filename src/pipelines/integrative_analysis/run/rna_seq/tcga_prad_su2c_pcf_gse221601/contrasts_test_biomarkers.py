import argparse
import functools
import json
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.integrative_analysis.utils import test_biomarkers
from r_wrappers.utils import map_gene_id
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
DESEQ2_PATH: Path = DATA_ROOT.joinpath("deseq2")
SAVE_PATH: Path = DATA_ROOT.joinpath("integrative_analysis").joinpath("test_biomarkers")
SAVE_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("HSPC", "PRIM"),
    ("MCRPC", "HSPC"),
]
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "PRIM": "#9ACD32",
    "HSPC": "#4A708B",
    "MCRPC": "#8B3A3A",
}
CONTRASTS_LEVELS_ORDER: Iterable[str] = ("PRIM", "HSPC", "MCRPC")
BIOMARKERS_FILE: Path = DATA_ROOT.joinpath("intersecting_ml_wgcna_genes.json")
BIOMARKERS_PLOT_TYPES: Dict[str, str] = {
    "DEGs (ML & WGCNA) in both contrasts": "violin",
    "DEGs (ML & WGCNA) only in prim/norm": "heatmap",
    "DEGs (ML & WGCNA) only in met/prim": "heatmap",
}
RANDOM_SEED: int = 8080
SPECIES: str = "Homo sapiens"
PARALLEL: bool = True

# 0. Setup
annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
annot_df_contrasts = deepcopy(
    annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin(contrast_conditions)]
)

vst_df = pd.read_csv(DESEQ2_PATH.joinpath(f"{exp_prefix}_vst.csv"), index_col=0)
org_db = OrgDB(SPECIES)
vst_df.index = map_gene_id(vst_df.index, org_db, "ENSEMBL", "SYMBOL")
# get rid of non-uniquely mapped transcripts
vst_df = vst_df.loc[~vst_df.index.str.contains("/", na=False)]
# remove all transcripts that share SYMBOL IDs
vst_df = vst_df.loc[vst_df.index.dropna().drop_duplicates(keep=False)]

with BIOMARKERS_FILE.open("r") as fp:
    BIOMARKERS = json.load(fp)

# 1. Collect inputs
input_collection = []
for alias, biomarkers in BIOMARKERS.items():
    input_collection.append(
        dict(
            biomarkers=biomarkers,
            contrast_factor=SAMPLE_CONTRAST_FACTOR,
            contrasts_levels=CONTRASTS_LEVELS,
            vst_df=vst_df,
            annot_df=deepcopy(annot_df_contrasts),
            exp_prefix=exp_prefix,
            deseq2_path=DESEQ2_PATH,
            save_path=SAVE_PATH,
            contrasts_levels_colors=CONTRASTS_LEVELS_COLORS,
            contrasts_levels_order=CONTRASTS_LEVELS_ORDER,
            plot_type=BIOMARKERS_PLOT_TYPES[alias],
            biomarkers_alias=alias,
            random_seed=RANDOM_SEED,
        )
    )


# 3. Run biomarkers tests
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=test_biomarkers),
            input_collection,
            threads=user_args["threads"] // 3,
        )
    else:
        for ins in tqdm(input_collection):
            test_biomarkers(**ins)
