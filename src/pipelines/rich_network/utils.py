import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx import from_pandas_adjacency
from sklearn.preprocessing import MinMaxScaler


def rich_wgcna_network(
    data_root: Path,
    exp_name: str,
    network_type: str,
    correlation_type: str,
    classifier_name: str,
    bootstrap_iterations: int = 10000,
    iterative: bool = False,
    corr_th: float = 0.7,
    remove_isolated: bool = False,
) -> None:
    """
    Generate a rich network including information from WGCNA (network topology and
        modules) and machine learning (shap values as node features).

    Args:
        data_root: Data root path.
        exp_name: Experiment name.
        org_db: Organism database.
        network_type: Type of the network to compute.
        correlation_type: Co-expression metric.
        classifier_name: Name of the classifier to use.
        corr_th: Correlation matrix threshold. Only edges with values higher than this
            are kept.
    """
    # 0. Setup
    rich_network_path = (
        data_root.joinpath("rich_network")
        .joinpath(exp_name)
        .joinpath("iterative" if iterative else "standard")
        .joinpath(classifier_name)
    )
    rich_network_path.mkdir(exist_ok=True, parents=True)
    wgcna_results_path = (
        data_root.joinpath("wgcna")
        .joinpath(exp_name)
        .joinpath("iterative" if iterative else "standard")
        .joinpath("results")
    )

    # 1. Load and process data
    # 1.1. WGCNA adjacency matrix
    wgcna_adj_mtx = pd.read_csv(
        wgcna_results_path.joinpath(
            f"{correlation_type}_{network_type}_adj_matrix.csv"
        ),
        index_col=0,
    )
    wgcna_net_genes = pd.read_csv(
        wgcna_results_path.joinpath(
            f"{correlation_type}_{network_type}_network_genes.csv"
        ),
    ).set_index("SYMBOL")
    network_nodes = wgcna_net_genes[
        wgcna_net_genes["ENTREZID"].isin(wgcna_adj_mtx.index)
    ]

    corr_matrix = wgcna_adj_mtx.values

    # Remove self-correlation
    np.fill_diagonal(corr_matrix, 0)

    # Threshold edges
    corr_matrix[np.abs(corr_matrix) < corr_th] = 0

    # Remove near-perfect correlation (likely FP)
    corr_matrix[np.abs(corr_matrix) > 0.99] = 0

    # Get graph
    wgcna_network_nx = from_pandas_adjacency(
        pd.DataFrame(
            corr_matrix, index=network_nodes.index, columns=network_nodes.index
        )
    )

    # 1.2. Get genes SHAP values
    shap_values = pd.read_csv(
        data_root.joinpath("ml_classifiers")
        .joinpath(exp_name)
        .joinpath(classifier_name)
        .joinpath("genes_features")
        .joinpath("bootstrap")
        .joinpath(f"bootstrap_{bootstrap_iterations}_shap_values.csv"),
    ).set_index("SYMBOL")

    shap_values["shap_value"] = MinMaxScaler().fit_transform(
        shap_values["shap_value"].to_numpy().reshape(-1, 1)
    )

    # 2. Define network
    # 2.1. Remove isolated nodes
    if remove_isolated:
        wgcna_network_nx.remove_nodes_from(list(nx.isolates(wgcna_network_nx)))

    # 2.2. Add SHAP values as node attributes
    nx.set_node_attributes(
        wgcna_network_nx, shap_values["shap_value"].to_dict(), "shap_value"
    )

    # 2.3. Add annotated fields as node attributes
    for annot_field, annot_values in wgcna_net_genes.items():
        nx.set_node_attributes(wgcna_network_nx, annot_values.to_dict(), annot_field)

    # 2.3.1. Remove all nodes belonging to M0
    wgcna_network_nx.remove_nodes_from(
        wgcna_net_genes[wgcna_net_genes["module"] == "M0"].index
    )

    # 2.3.2. Remove all nodes matching certain gene names
    unwanted_traits = (
        "uncharacterized",
        "microrna",
        "long intergenic non-protein coding",
    )
    unwanted_symbol = (
        "LOC",
        "MIR",
        "LINC",
    )
    wgcna_network_nx.remove_nodes_from(
        wgcna_net_genes[
            (wgcna_net_genes["GENENAME"].str.lower().str.startswith(unwanted_traits))
            | (wgcna_net_genes.index.str.startswith(unwanted_symbol))
        ].index
    )

    # 3. Export network to disk
    if nx.is_empty(wgcna_network_nx):
        logging.info("Network is empty (has no edges), nothing to save.")
        return

    corr_th_str = str(corr_th).replace(".", "_")
    remove_isolated_str = "no_isolates" if remove_isolated else "full"

    nx.write_gexf(
        wgcna_network_nx,
        rich_network_path.joinpath(
            f"{exp_name}_corr_th_{corr_th_str}_{remove_isolated_str}_wgcna_ml_net.gexf"
        ),
    )

    nx.write_graphml(
        wgcna_network_nx,
        rich_network_path.joinpath(
            f"{exp_name}_corr_th_{corr_th_str}_{remove_isolated_str}_"
            "wgcna_ml_net.graphml"
        ),
    )
