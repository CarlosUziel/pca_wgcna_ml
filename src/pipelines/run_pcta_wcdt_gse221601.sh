#!/bin/bash

# This script runs all the scripts from the PCTA-WCDT-GSE221601 pipeline

# pcta_wcdt_gse221601
# Differential Expression Analysis
python differential_expression/run/pcta_wcdt_gse221601/contrasts.py
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts.py

# WGCNA Analysis
python wgcna/run/rna_seq/pcta_wcdt_gse221601/contrasts.py
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts_wgcna.py

# ML Analysis
python ml_classifiers/run/rna_seq/pcta_wcdt_gse221601/contrasts_genes_features_hptuning.py
python ml_classifiers/run/rna_seq/pcta_wcdt_gse221601/contrasts_genes_features_bootstrap_training.py
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts_ml.py

# Integrative Analysis
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts_intersect_degs.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts_intersect_pathways.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts_intersect_pathways_genes.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts_intersect_wgcna.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601/contrasts_intersect_wgcna_pathways.py

# pcta_wcdt_gse221601_lrt
# Differential Expression Analysis
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601_lrt/lrt.py

# WGCNA Analysis
python wgcna/run/rna_seq/pcta_wcdt_gse221601_lrt/lrt.py
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601_lrt/lrt_wgcna.py

# pcta_wcdt_gse221601_filtered

# Differential Expression Analysis
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts.py

# WGCNA Analysis
python wgcna/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts.py
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_wgcna.py

# Integrative Analysis
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_intersect_degs.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_intersect_pathways.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_intersect_pathways_genes.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_intersect_wgcna.py
python integrative_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_intersect_wgcna_pathways.py

# ML Analysis
python ml_classifiers/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_genes_features_hptuning.py
python ml_classifiers/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_genes_features_bootstrap_training.py
python functional_analysis/run/rna_seq/pcta_wcdt_gse221601_filtered/contrasts_ml.py