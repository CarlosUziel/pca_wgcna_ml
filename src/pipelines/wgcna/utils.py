import json
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import seaborn as sns
from matplotlib import pyplot as plt

from components.functional_analysis.orgdb import OrgDB
from data.utils import filter_genes_wrt_annotation
from r_wrappers.utils import map_gene_id, rpy2_df_to_pd_df, save_rds
from r_wrappers.wgcna import (
    adjacency,
    blockwise_modules,
    blockwise_modules_iterative,
    choose_top_hub_in_each_module,
    enable_threads,
    labels2colors,
    network_concepts,
    pick_soft_threshold,
    plot_dendro_and_colors,
)


def differential_expression(
    data_file: Path,
    wgcna_path: Path,
    degs_file: Path,
    annot_df: pd.DataFrame,
    contrast_factor: str,
    contrast: Tuple[str, str],
    org_db: OrgDB,
    network_type: str = "signed",
    correlation_type: str = "bicor",
    threads: int = 8,
    iterative: bool = False,
) -> None:
    """Run WGCNA using differentially expressed genes (DEGs) as input.

    Args:
        data_file: Path to gene expression data file (e.g., VST from DESeq2)
        wgcna_path: Pipeline root directory.
        degs_file: Path to DE results (filtered).
        annot_df: Samples annotation dataframe.
        contrast_factor: Annotation field by which sample classes are determined.
        contrast: Contrast DEGs come from.
        org_db: Organism database.
        network_type: Type of the network to compute.
        correlation_type: Co-expression metric.
        threads: WGCNA threads.
        iterative: whether to run iterativeWGCNA instead.
    """
    results_path: Path = wgcna_path.joinpath("results")
    results_path.mkdir(exist_ok=True, parents=True)

    plots_path: Path = wgcna_path.joinpath("plots")
    plots_path.mkdir(exist_ok=True, parents=True)

    ####################################################################################
    # 1. Process expression data
    # 1.1. Load expression data and annotation
    vst_df = pd.read_csv(data_file, index_col=0)
    degs_df = pd.read_csv(degs_file, index_col=0, dtype={"ENTREZID": str})

    # 1.2. Keep only conditions in contrasts
    annot_df_contrasts = deepcopy(
        annot_df.loc[annot_df[contrast_factor].isin(contrast), :]
    )
    vst_df = vst_df.loc[
        degs_df.index, vst_df.columns.intersection(annot_df_contrasts.index)
    ]
    annot_df_contrasts = annot_df_contrasts.loc[vst_df.columns, :]

    # 1.3. Annotate genes and filter according to gene names
    vst_df = vst_df.loc[filter_genes_wrt_annotation(vst_df.index, org_db), :]
    vst_df.index = map_gene_id(vst_df.index, org_db, "ENSEMBL", "ENTREZID")
    # get rid of non-uniquely mapped transcripts
    vst_df = vst_df.loc[~vst_df.index.str.contains("/", na=False)]
    # remove all transcripts that share ENTREZIDs IDs
    vst_df = vst_df.loc[vst_df.index.dropna().drop_duplicates(keep=False)]

    ####################################################################################
    # 2. Compute WGCNA network
    enable_threads(threads)
    vst_net = vst_df.transpose()
    vst_net.to_csv(wgcna_path.joinpath("vst_net.csv"))

    # 2.1. Get network concepts
    net_concepts_file = wgcna_path.joinpath(f"{network_type}_network_concepts.RDS")

    logging.info("Calculating network concepts...")
    net_concepts = network_concepts(vst_net, power=2, networkType=network_type)
    save_rds(net_concepts, net_concepts_file)

    # 2.1.2. Save genes' clustering coefficient and connectivity statistics
    gene_stats_df = pd.DataFrame(
        [
            np.array(net_concepts.rx2("ClusterCoef").rx(ro.StrVector(vst_net.columns))),
            np.array(
                net_concepts.rx2("Connectivity").rx(ro.StrVector(vst_net.columns))
            ),
        ],
        index=["ClusterCoef", "Connectivity"],
        columns=vst_net.columns,
    ).transpose()

    # 2.2. Estimate soft threshold (power)
    sft_file = wgcna_path.joinpath(f"{correlation_type}_{network_type}_sft.RDS")

    logging.info("Calculating soft power threshold...")
    sft = pick_soft_threshold(
        vst_net,
        corFnc=correlation_type,
        RsquaredCut=0.8,
        networkType=network_type,
        verbose=0,
    )
    save_rds(sft, sft_file)

    power = sft.rx2("powerEstimate")[0]
    if (type(power) == ro.NA_Real) or not (1 <= power <= 30):
        power = 6 if network_type == "unsigned" else 12
    logging.info(f"Power estimate chosen: {power}")

    # 2.2.1. Save power estimate to disk
    with results_path.joinpath(f"{correlation_type}_{network_type}_power.json").open(
        "w"
    ) as fp:
        json.dump(
            power,
            fp,
            indent=4,
            sort_keys=True,
        )

    # 2.3. Build the network
    network_file = wgcna_path.joinpath(f"{correlation_type}_{network_type}_network.RDS")

    logging.info("Calculating network...")
    wgcna_args = dict(
        networkType=network_type,
        corType=correlation_type,
        power=power,
        maxBlockSize=35000,
        minModuleSize=30,
        reassignThreshold=1e-6,
        detectCutHeight=0.998,
        mergeCutHeight=0.15,
        deepSplit=2,
        numericLabels=True,
        pamStage=True,
        pamRespectsDendro=True,
        verbose=0,
    )
    if iterative:
        network = blockwise_modules_iterative(vst_net, **wgcna_args)
    else:
        network = blockwise_modules(vst_net, **wgcna_args)
    save_rds(network, network_file)

    # 2.3.1. Save genes per module
    module_genes = defaultdict(list)
    for module_id, gene in zip(network.rx2("colors"), network.rx2("colors").names):
        module_genes[f"M{int(module_id)}"].append(gene)

    gene_stats_annotations = (
        pd.concat(
            [
                pd.Series(
                    {
                        gene: module_name
                        for module_name, gene_list in module_genes.items()
                        for gene in gene_list
                    },
                    name="module",
                ).to_frame(),
                gene_stats_df,
                degs_df[degs_df["ENTREZID"].isin(vst_net.columns)].set_index(
                    "ENTREZID"
                ),
            ],
            axis=1,
        )
        .rename_axis("ENTREZID")
        .sort_values(["ClusterCoef", "Connectivity"], ascending=False)
    )

    gene_stats_annotations.to_csv(
        results_path.joinpath(f"{correlation_type}_{network_type}_network_genes.csv")
    )

    for module_name, gene_list in module_genes.items():
        gene_stats_annotations.loc[gene_list, :].sort_values(
            ["ClusterCoef", "Connectivity"], ascending=False
        ).to_csv(
            results_path.joinpath(
                f"{correlation_type}_{network_type}_"
                + f"{module_name}_{len(gene_list)}_genes.csv"
            )
        )

    # 2.3.2. Get and save gene hubs
    module_hub_genes = choose_top_hub_in_each_module(
        vst_net, network.rx2("colors"), type=network_type, power=power
    )

    hub_genes = list(module_hub_genes)
    hub_genes_annotations = gene_stats_annotations.loc[hub_genes, :].reset_index()
    hub_genes_annotations.index = [f"M{m}" for m in module_hub_genes.names]
    hub_genes_annotations.to_csv(
        results_path.joinpath(f"{correlation_type}_{network_type}_hub_genes.csv")
    )

    # 2.3.3. Get adjacency matrix and save
    adj_matrix = adjacency(
        vst_net,
        type=network_type,
        power=power,
        corFnc=correlation_type,
    )
    adj_matrix_df = rpy2_df_to_pd_df(adj_matrix)
    adj_matrix_df.index = vst_net.columns
    adj_matrix_df.columns = vst_net.columns
    adj_matrix_df.to_csv(
        results_path.joinpath(f"{correlation_type}_{network_type}_adj_matrix.csv")
    )

    ####################################################################################
    # 3. Calculate association between modules and traits
    mod_eigengenes_df = rpy2_df_to_pd_df(network.rx2("MEs"))
    mod_eigengenes_df.columns = [
        int(c.replace("ME", "")) for c in mod_eigengenes_df.columns
    ]
    mod_eigengenes_df.sort_index(axis=1, inplace=True)

    # drop module #0
    mod_eigengenes_df.drop(columns=[0], inplace=True)

    module_associations = {}
    for condition in contrast:
        condition_encoding = np.array(
            [
                1 if sample_cluster == condition else 0
                for sample_cluster in annot_df_contrasts[contrast_factor]
            ]
        )
        module_associations[condition] = {
            module_name: np.corrcoef(
                mod_eigengenes_df[module_name].to_numpy(), condition_encoding
            )[0, 1]
            for module_name in mod_eigengenes_df.columns
        }
    module_associations_df = pd.DataFrame(module_associations)
    module_associations_df.sort_index(inplace=True)

    # 3.2.1. Save results
    module_associations_df.to_csv(
        results_path.joinpath(
            f"{correlation_type}_{network_type}_" + "modules_contrast_correlations.csv"
        )
    )

    ####################################################################################
    # 4. Plotting
    # 4.1. Dendogram
    save_path = plots_path.joinpath(f"{correlation_type}_{network_type}_dendrogram.pdf")
    plot_dendro_and_colors(
        network.rx2("dendrograms")[0],
        labels2colors(network.rx2("colors").rx(network.rx2("goodGenes"))),
        save_path,
        groupLabels=ro.StrVector(["modules"]),
    )

    # 4.2. Eigengenes
    plt.figure(figsize=(10, 10), dpi=300)
    sns.violinplot(data=mod_eigengenes_df.melt(), x="variable", y="value")
    plt.title("Samples eigengenes plot")
    plt.xlabel("WGCNA Module")
    plt.ylabel("Eigengene values")
    plt.savefig(
        plots_path.joinpath(
            f"{correlation_type}_{network_type}_" + "samples_eigengenes_plot.pdf"
        )
    )

    # 4.3. Correlation heatmap
    plt.figure(figsize=(10, 10), dpi=300)
    sns.set_theme()
    sns.heatmap(module_associations_df, cmap="Blues")
    plt.title("Module-Condition correlation")
    plt.ylabel("WGCNA Module")
    plt.yticks(rotation=0)
    plt.xlabel(f"Conditions ({contrast_factor})")
    plt.savefig(
        plots_path.joinpath(
            f"{correlation_type}_{network_type}_" + "module_condition_corr_heatmap.pdf"
        )
    )


def differential_methylation(
    meth_values_file: Path,
    wgcna_path: Path,
    custom_meth_probes_file: Path,
    annot_df: pd.DataFrame,
    contrast_factor: str,
    contrast: Tuple[str, str],
    org_db: OrgDB,
    network_type: str = "signed",
    correlation_type: str = "bicor",
    threads: int = 8,
    iterative: bool = False,
) -> None:
    """Run WGCNA using differentially methylated probes (DMPs) as input.

    Args:
        meth_values_file: A .csv file containing either methylation B-values or M-values
            with shape [#probes, #samples]
        wgcna_path: Pipeline root directory.
        custom_meth_probes_file: A .csv file with pre-selected differentially methylated
            probes, annotated to gene regions.
        annot_df: Samples annotation dataframe.
        contrast_factor: Annotation field by which sample classes are determined.
        contrast: Contrast DEGs come from.
        org_db: Organism database.
        network_type: Type of the network to compute.
        correlation_type: Co-expression metric.
        threads: WGCNA threads.
        iterative: whether to run iterativeWGCNA instead
    """
    results_path: Path = wgcna_path.joinpath("results")
    results_path.mkdir(exist_ok=True, parents=True)

    plots_path: Path = wgcna_path.joinpath("plots")
    plots_path.mkdir(exist_ok=True, parents=True)

    ####################################################################################
    # 1. Process expression data
    # 1.1. Load expression data and annotation
    meth_values_df = pd.read_csv(meth_values_file, index_col=0)
    custom_meth_probes_df = pd.read_csv(custom_meth_probes_file, index_col=0).dropna(
        subset="annot.gene_id"
    )
    custom_meth_probes_df["annot.gene_id"] = (
        custom_meth_probes_df["annot.gene_id"].astype(int).astype(str)
    )

    # 1.2. Keep only conditions in contrasts
    annot_df_contrasts = deepcopy(
        annot_df.loc[annot_df[contrast_factor].isin(contrast), :]
    )
    meth_values_df = meth_values_df.loc[
        custom_meth_probes_df.index,
        meth_values_df.columns.intersection(annot_df_contrasts.index),
    ]
    annot_df_contrasts = annot_df_contrasts.loc[meth_values_df.columns, :]

    # 1.3. Filter according to custom significant probes
    valid_genes = filter_genes_wrt_annotation(
        custom_meth_probes_df["annot.gene_id"].drop_duplicates(keep=False),
        org_db,
        "ENTREZID",
    )
    probes = custom_meth_probes_df[
        custom_meth_probes_df["annot.gene_id"].isin(valid_genes)
    ].index

    meth_values_df = meth_values_df.loc[probes, :]

    ####################################################################################
    # 2. Compute WGCNA network
    enable_threads(threads)
    meth_values_net = meth_values_df.transpose()
    meth_values_net.to_csv(wgcna_path.joinpath("meth_values_net.csv"))

    # 2.1. Get network concepts
    net_concepts_file = wgcna_path.joinpath(f"{network_type}_network_concepts.RDS")

    logging.info("Calculating network concepts...")
    net_concepts = network_concepts(meth_values_net, power=2, networkType=network_type)
    save_rds(net_concepts, net_concepts_file)

    # 2.1.2. Save probes' clustering coefficient and connectivity statistics
    probes_stats_df = pd.DataFrame(
        [
            np.array(
                net_concepts.rx2("ClusterCoef").rx(
                    ro.StrVector(meth_values_net.columns.astype(str))
                )
            ),
            np.array(
                net_concepts.rx2("Connectivity").rx(
                    ro.StrVector(meth_values_net.columns.astype(str))
                )
            ),
        ],
        index=["ClusterCoef", "Connectivity"],
        columns=meth_values_net.columns.astype(str),
    ).transpose()
    probes_stats_annot_df = pd.concat(
        [
            probes_stats_df,
            custom_meth_probes_df.loc[probes_stats_df.index, :][
                [
                    "annot.seqnames",
                    "annot.start",
                    "annot.end",
                    "annot.width",
                    "annot.strand",
                    "annot.id",
                    "annot.tx_id",
                    "annot.gene_id",
                    "annot.symbol",
                    "annot.type",
                ]
            ],
        ],
        axis=1,
    ).sort_values(["ClusterCoef", "Connectivity"], ascending=False)
    probes_stats_annot_df.to_csv(
        results_path.joinpath(f"{correlation_type}_{network_type}_network_probes.csv")
    )

    # 2.2. Estimate soft threshold (power)
    sft_file = wgcna_path.joinpath(f"{correlation_type}_{network_type}_sft.RDS")

    logging.info("Calculating soft power threshold...")
    sft = pick_soft_threshold(
        meth_values_net,
        corFnc=correlation_type,
        RsquaredCut=0.8,
        networkType=network_type,
        verbose=0,
    )
    save_rds(sft, sft_file)

    power = sft.rx2("powerEstimate")[0]
    if (type(power) == ro.NA_Real) or not (1 <= power <= 30):
        power = 6 if network_type == "unsigned" else 12
    logging.info(f"Power estimate chosen: {power}")

    # 2.2.1. Save power estimate to disk
    with results_path.joinpath(f"{correlation_type}_{network_type}_power.json").open(
        "w"
    ) as fp:
        json.dump(
            power,
            fp,
            indent=4,
            sort_keys=True,
        )

    # 2.3. Build the network
    network_file = wgcna_path.joinpath(f"{correlation_type}_{network_type}_network.RDS")

    logging.info("Calculating network...")
    wgcna_args = dict(
        networkType=network_type,
        corType=correlation_type,
        power=power,
        maxBlockSize=35000,
        minModuleSize=30,
        reassignThreshold=1e-6,
        detectCutHeight=0.998,
        mergeCutHeight=0.15,
        deepSplit=2,
        numericLabels=True,
        pamStage=True,
        pamRespectsDendro=True,
        verbose=0,
    )
    if iterative:
        network = blockwise_modules_iterative(meth_values_net, **wgcna_args)
    else:
        network = blockwise_modules(meth_values_net, **wgcna_args)
    save_rds(network, network_file)

    # 2.3.1. Save probes per module
    module_probes = defaultdict(list)
    for module_id, probe in zip(network.rx2("colors"), network.rx2("colors").names):
        module_probes[f"M{int(module_id)}"].append(probe)

    pd.Series(
        {
            probe: module_name
            for module_name, probe_list in module_probes.items()
            for probe in probe_list
        },
        name="module",
    ).to_csv(
        results_path.joinpath(
            f"{correlation_type}_{network_type}_" + "probe_modules.csv"
        )
    )

    for module_name, probes_list in module_probes.items():
        probes_stats_annot_df.loc[probes_list, :].sort_values(
            ["ClusterCoef", "Connectivity"], ascending=False
        ).to_csv(
            results_path.joinpath(
                f"{correlation_type}_{network_type}_"
                + f"{module_name}_{len(probes_list)}_probes.csv"
            )
        )

    # 2.3.2. Get and save probe hubs
    module_hub_probes = choose_top_hub_in_each_module(
        meth_values_net, network.rx2("colors"), type=network_type, power=power
    )

    hub_probes = list(module_hub_probes)
    hub_probes_annotations = probes_stats_annot_df.loc[hub_probes, :].reset_index()
    hub_probes_annotations.index = [f"M{m}" for m in module_hub_probes.names]
    hub_probes_annotations.to_csv(
        results_path.joinpath(f"{correlation_type}_{network_type}_hub_probes.csv")
    )

    # 2.3.3. Get adjacency matrix and save
    adj_matrix = adjacency(
        meth_values_net,
        type=network_type,
        power=power,
        corFnc=correlation_type,
    )
    adj_matrix_df = rpy2_df_to_pd_df(adj_matrix)
    adj_matrix_df.index = meth_values_net.columns
    adj_matrix_df.columns = meth_values_net.columns
    adj_matrix_df.to_csv(
        results_path.joinpath(f"{correlation_type}_{network_type}_adj_matrix.csv")
    )

    ####################################################################################
    # 3. Calculate association between modules and traits
    mod_eigenprobes_df = rpy2_df_to_pd_df(network.rx2("MEs"))
    mod_eigenprobes_df.columns = [
        int(c.replace("ME", "")) for c in mod_eigenprobes_df.columns
    ]
    mod_eigenprobes_df.sort_index(axis=1, inplace=True)

    module_associations = {}

    for condition in contrast:
        condition_encoding = np.array(
            [
                1 if sample_cluster == condition else 0
                for sample_cluster in annot_df_contrasts[contrast_factor]
            ]
        )
        module_associations[condition] = {
            module_name: np.corrcoef(
                mod_eigenprobes_df[module_name].to_numpy(), condition_encoding
            )[0, 1]
            for module_name in mod_eigenprobes_df.columns
        }
    module_associations_df = pd.DataFrame(module_associations)
    module_associations_df.sort_index(inplace=True)

    # 3.2.1. Save results
    module_associations_df.to_csv(
        results_path.joinpath(
            f"{correlation_type}_{network_type}_" + "modules_contrast_correlations.csv"
        )
    )

    ####################################################################################
    # 4. Plotting
    # 4.1. Dendogram
    save_path = plots_path.joinpath(f"{correlation_type}_{network_type}_dendrogram.pdf")
    plot_dendro_and_colors(
        network.rx2("dendrograms")[0],
        labels2colors(network.rx2("colors").rx(network.rx2("goodGenes"))),
        save_path,
        groupLabels=ro.StrVector(["modules"]),
    )

    # 4.2. Eigenprobes
    plt.figure(figsize=(10, 10), dpi=300)
    sns.violinplot(data=mod_eigenprobes_df.melt(), x="variable", y="value")
    plt.title("Samples eigenprobes plot")
    plt.xlabel("WGCNA Module")
    plt.ylabel("Eigenprobe values")
    plt.savefig(
        plots_path.joinpath(
            f"{correlation_type}_{network_type}_" + "samples_eigenprobes_plot.pdf"
        )
    )

    # 4.3. Correlation heatmap
    plt.figure(figsize=(10, 10), dpi=300)
    sns.set_theme()
    sns.heatmap(module_associations_df, cmap="Blues")
    plt.title("Module-Condition correlation")
    plt.ylabel("WGCNA Module")
    plt.yticks(rotation=0)
    plt.xlabel(f"Conditions ({contrast_factor})")
    plt.savefig(
        plots_path.joinpath(
            f"{correlation_type}_{network_type}_" + "module_condition_corr_heatmap.pdf"
        )
    )
