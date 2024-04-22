import argparse
import functools
import json
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import filter_df, parallelize_map
from pipelines.wgcna.utils import differential_methylation
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
GOI_ENSEMBL: str = "ENSG00000086205"  # ENSEMBL ID for FOLH1 (PSMA)
SPECIES: str = "Homo sapiens"
org_db = OrgDB(SPECIES)
GOI_SYMBOL = map_gene_id([GOI_ENSEMBL], org_db, "ENSEMBL", "SYMBOL")[0]
DATA_ROOT: Path = STORAGE.joinpath(f"TCGA_PRAD_SU2C_RNASeq_{GOI_SYMBOL}")
WGCNA_ROOT = DATA_ROOT.joinpath("wgcna")
WGCNA_ROOT.mkdir(exist_ok=True, parents=True)
ANNOT_PATH: Path = DATA_ROOT.joinpath("data").joinpath(
    f"samples_annotation_common_{GOI_SYMBOL}.csv"
)
GENOME: str = "hg38"
SAMPLE_CONTRAST_FACTOR: str = "sample_type"

GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
PERCENTILES: Iterable[int] = (10, 15, 20)
WITHIN_SAMPLES: Iterable[str] = ("prim",)
WITHIN_SAMPLE_CONTRAST_FILTERS = [
    (
        {
            SAMPLE_CONTRAST_FACTOR: (sample_type,),
            f"{GOI_LEVEL_PREFIX}_{percentile}": ("high",),
        },
        {
            SAMPLE_CONTRAST_FACTOR: (sample_type,),
            f"{GOI_LEVEL_PREFIX}_{percentile}": ("low",),
        },
    )
    for sample_type in WITHIN_SAMPLES
    for percentile in PERCENTILES
    if not (sample_type == "met" and percentile != 20)
]
ASYMETRIC_SAMPLE_CONTRAST: Iterable[Tuple[Tuple[str, str]]] = (
    (("prim", "high"), ("norm", None)),
    (("prim", "low"), ("norm", None)),
)
ASYMETRIC_SAMPLE_CONTRAST_FILTERS = [
    (
        {
            SAMPLE_CONTRAST_FACTOR: (test[0],),
            f"{GOI_LEVEL_PREFIX}_{percentile}": (test[1],),
        },
        {
            SAMPLE_CONTRAST_FACTOR: (control[0],),
        },
    )
    for test, control in ASYMETRIC_SAMPLE_CONTRAST
    for percentile in PERCENTILES
]
SAMPLE_CLUSTER_CONTRAST_LEVELS: Iterable = (
    WITHIN_SAMPLE_CONTRAST_FILTERS + ASYMETRIC_SAMPLE_CONTRAST_FILTERS
)
with DATA_ROOT.joinpath("SAMPLE_CLUSTER_CONTRAST_LEVELS.json").open("w") as fp:
    json.dump(SAMPLE_CLUSTER_CONTRAST_LEVELS, fp, indent=True)
NORM_TYPES: Iterable[str] = ("noob_quantile",)
GENE_ANNOTS: Iterable[str] = (f"{GENOME}_genes_promoters",)
N_THREADS: int = 4
P_COLS: Iterable[str] = ("P.Value", "adj.P.Val")
P_THS: Iterable[float] = (0.05, 0.01)
LFC_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
LFC_THS: Iterable[float] = (0.0, 1.0, 2.0)
MEAN_METH_DIFF_THS: Iterable[float] = (0.1, 0.2, 0.3)
SPECIES: str = "Homo sapiens"
NETWORK_TYPES: Iterable[str] = ("signed",)
CORRELATION_TYPES: Iterable[str] = ("bicor",)
ITERATIVE_MODES: Iterable[bool] = (True, False)
WGCNA_THREADS: int = 4
PARALLEL: bool = True

annot_df = (
    pd.read_csv(ANNOT_PATH, dtype=str)
    .rename(columns={"barcode": "Basename"})
    .sort_values("probe")
    .drop_duplicates("Basename")
)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))


input_collection = []
for sample_cluster_contrast in SAMPLE_CLUSTER_CONTRAST_LEVELS:
    # 0. Setup
    test_filters, control_filters = sample_cluster_contrast

    # 1. Multi-level samples annotation
    annot_df_contrasts = deepcopy(annot_df)

    # Keep only same sample type
    if (sample_type := test_filters[SAMPLE_CONTRAST_FACTOR][0]) == control_filters[
        SAMPLE_CONTRAST_FACTOR
    ][0]:
        annot_df_contrasts = annot_df_contrasts[
            annot_df_contrasts[SAMPLE_CONTRAST_FACTOR] == sample_type
        ]

    # 1.1. Annotation of test samples
    goi_level = [key for key in test_filters.keys() if GOI_LEVEL_PREFIX in key]
    contrast_level_test = "_".join(chain(*test_filters.values())) + (
        goi_level[0].replace(GOI_LEVEL_PREFIX, "") if len(goi_level) > 0 else ""
    )
    annot_df_contrasts.loc[
        filter_df(annot_df_contrasts, test_filters).index, GOI_LEVEL_PREFIX
    ] = contrast_level_test

    # 1.2. Annotation of control samples
    goi_level = [key for key in control_filters.keys() if GOI_LEVEL_PREFIX in key]
    contrast_level_control = "_".join(chain(*control_filters.values())) + (
        goi_level[0].replace(GOI_LEVEL_PREFIX, "") if len(goi_level) > 0 else ""
    )
    annot_df_contrasts.loc[
        filter_df(annot_df_contrasts, control_filters).index, GOI_LEVEL_PREFIX
    ] = contrast_level_control

    # 1.3. Set experiment prefix and remove unnecesary samples
    contrasts_levels = (contrast_level_test, contrast_level_control)
    annot_df_contrasts = pd.concat(
        (
            annot_df_contrasts[
                annot_df_contrasts[GOI_LEVEL_PREFIX] == contrast_level_test
            ],
            annot_df_contrasts[
                annot_df_contrasts[GOI_LEVEL_PREFIX] == contrast_level_control
            ],
        )
    )

    for (
        norm_type,
        gene_annot,
        mean_meth_diff_th,
    ) in product(NORM_TYPES, GENE_ANNOTS, MEAN_METH_DIFF_THS):
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        exp_prefix = (
            f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
            f"diff_meth_probes_{norm_type}_top_table_"
            f"{contrast_level_test}_vs_{contrast_level_control}"
        )

        for (
            p_col,
            p_th,
            lfc_level,
            lfc_th,
            mean_meth_diff_th,
            network_type,
            correlation_type,
            iterative,
        ) in product(
            P_COLS,
            P_THS,
            LFC_LEVELS,
            LFC_THS,
            MEAN_METH_DIFF_THS,
            NETWORK_TYPES,
            CORRELATION_TYPES,
            ITERATIVE_MODES,
        ):
            p_col_str = p_col.replace(".", "_")
            p_thr_str = str(p_th).replace(".", "_")
            lfc_thr_str = str(lfc_th).replace(".", "_")
            mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
            exp_name = (
                f"{exp_prefix}_sig_{p_col_str}_{p_thr_str}_{lfc_level}_{lfc_thr_str}_"
                f"wrt_mean_diff_{mean_meth_diff_th_str}_{gene_annot}"
            )

            custom_meth_probes_file = DATA_ROOT.joinpath("minfi").joinpath(
                f"{exp_name}.csv"
            )
            if not custom_meth_probes_file.exists():
                continue

            input_collection.append(
                dict(
                    meth_values_file=(
                        DATA_ROOT.joinpath("minfi").joinpath(
                            f"{exp_prefix}_"
                            f"diff_meth_probes_{norm_type}_"
                            f"{contrast_level_test}_vs_{contrast_level_control}"
                            f"b_values_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
                        )
                    ),
                    wgcna_path=(
                        WGCNA_ROOT.joinpath(exp_name).joinpath(
                            "iterative" if iterative else "standard"
                        )
                    ),
                    custom_meth_probes_file=custom_meth_probes_file,
                    annot_df=annot_df_contrasts,
                    contrast_factor=GOI_LEVEL_PREFIX,
                    contrast=(contrast_level_test, contrast_level_control),
                    org_db=org_db,
                    network_type=network_type,
                    correlation_type=correlation_type,
                    threads=WGCNA_THREADS,
                    iterative=iterative,
                )
            )


if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=differential_methylation),
            input_collection,
            threads=60 // WGCNA_THREADS,
        )
    else:
        for ins in tqdm(input_collection):
            differential_methylation(**ins)
