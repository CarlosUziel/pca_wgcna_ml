######################

library(cola)
library(ComplexHeatmap)
library(circlize)
library(GetoptLong)
library("simplifyEnrichment")
library("AnnotationDbi")
library("org.Hs.eg.db")
library(cowplot)
library(NMF)

##################on up and down regulated separately
up1 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)

up2 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)

down1 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_down_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)

down2 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_down_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)

all1 <- rbind(up1, down1)

all2 <- rbind(up2, down2)

#############only for DEGS from prim vs norm
dds2 <- dds[,(colData(dds)$sample_type %in% c("prim","norm"))]

select_genes<-rownames(subset(all2))
ddsresSig<-(dds2)[select_genes,]

mat1 <- (assay(ddsresSig))

mat_subset <- subset(mat1, !is.na(rownames(mat1)))
#head(mat_subset)
nrow(mat_subset)

subtype = (colData(dds2)) 
head(subtype)

subtype = (colData(dds2)[, 2, drop=F])
head(subtype)

subtype2 = t(subtype) 
subtype3 = structure(unlist(subtype2[1, -(1:1)]), names = colnames(subtype2)[-(1:1)])
head(subtype3)
#TCGA-XK-AAJR-01A TCGA-KC-A4BV-01A TCGA-KC-A7F6-01A TCGA-YL-A8SI-01A 
#            prim             prim             prim             prim 
#TCGA-M7-A71Z-01A TCGA-ZG-A8QW-01A 
#            prim             prim 
#Levels: met norm prim
#
subtype_col = structure(seq_len(3), names = levels(subtype3))
head(subtype_col)
# met norm prim 
#   1    2    3 
###########removed genes with zero variance
mat = mat_subset[, names(subtype3)]
mat = adjust_matrix(mat)

###run_all_consensus_partition_methods
rlall2 = run_all_consensus_partition_methods(
    mat,
  #  top_value_method = c("SD", "MAD", 'CV', 'ATC'),
  #  partition_method = c("hclust", "kmeans", "skmeans", "pam", "mclust),
    mc.cores = 4,
    anno = data.frame(subtype = subtype3),
    anno_col = list(subtype = subtype_col)
)
saveRDS(rlall2, file = "resSig_Prim_vs_Norm_all_DEGs_dds_cleaned_PRAD_DbGAP_03.rds")

rlall2 = readRDS("resSig_Prim_vs_Norm_all_DEGs_dds_cleaned_PRAD_DbGAP_03.rds")

################
res = rlall2["CV:hclust"]
set.seed(123)
#res = golub_cola["ATC:skmeans"]
df = get_signatures(res, k = 2, row_km = 2, plot = FALSE)
write.table(df, file="result_top_signatures_CVhclust_k2_km2_all_DEGs_primvsnorm.tsv", quote=F, sep="\t")
cl = get_classes(res, k = 2)[, 1]
cl[cl == 1] = 4; cl[cl == 2] = 1; cl[cl == 4] = 2
m = get_matrix(res)[df$which_row, ]
m = t(scale(t(m)))
nrow(m)
#[1] 3093
write.table(m, file="top_signatures_CVhclust_k2_km2_all_DEGs_primvsnorm.tsv", quote=F, sep="\t")

ht = Heatmap(m, name = "Scaled\nexpression",
col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
top_annotation = HeatmapAnnotation(
sample_class = cl,
sample_type = get_anno(res)[, 1],
col = list(sample_class = cola:::brewer_pal_set2_col[1:3],
sample_type = c("prim"="#4A708B", "norm"="#9ACD32"))),
#subtype = c("Primary"="#4A708B", "Metastatic"="#8B3A3A"))),
     #  subtype = get_anno_col(res)[[1]])),
row_split = paste0("cluster", df$km), show_row_dend = FALSE, show_row_names = FALSE,
column_split = cl, show_column_dend = FALSE, show_column_names = FALSE,
column_title = qq("1) @{nrow(df)} signature genes under Padj < 0.05")
)
p1 = grid.grabExpr(draw(ht, merge_legend = TRUE, padding = unit(c(12, 2, 2, 2), "mm")))
lt = functional_enrichment(res, k = 2)
ago = c(rownames(lt[[1]]), rownames(lt[[2]]))
ago = unique(ago)
pm = matrix(1, nrow = length(ago), ncol = 2)
rownames(pm) = ago
colnames(pm) = c("cluster1", "cluster2")
              
pm[lt[[1]]$ID, 1] = lt[[1]]$p.adjust
pm[lt[[2]]$ID, 2] = lt[[2]]$p.adjust
fdr_cutoff = 0.05
pm = pm[apply(pm, 1, function(x) any(x < fdr_cutoff)), ]
write.table(pm, file="GO_clusters_CVhclust_k2_km2_all_DEGs_primvsnorm.tsv", quote=F, sep="\t")

names(lt)
head(lt[[1]][, 1:7])
write.table((lt[[1]][, 1:7]), file="BP_km1_clusters_CVhclust_k2_km2_all_DEGs_primvsnorm.tsv", quote=F, sep="\t")
head(lt[[2]][, 1:7])
write.table((lt[[2]][, 1:7]), file="BP_km2_clusters_CVhclust_k2_km2_all_DEGs_primvsnorm.tsv", quote=F, sep="\t")

all_go_id = rownames(pm)
#library(simplifyEnrichment)
sim_mat = GO_similarity(all_go_id)
col_fun_p = colorRamp2(c(0, -log10(fdr_cutoff), 4), c("#EEE8AA", "#F7F7F7", "#C51B7D"))
ht_fdr = Heatmap(-log10(pm), col = col_fun_p, name = "Padj",
show_row_names = FALSE, cluster_columns = FALSE,
border = "black",
heatmap_legend_param = list(at = c(0, -log10(fdr_cutoff), 4), 
labels = c("1", fdr_cutoff, "< 0.0001")),
width = unit(1.5, "cm"), use_raster = TRUE)
p2 = grid.grabExpr(
simplifyGO(sim_mat, ht_list = ht_fdr, word_cloud_grob_param = list(max_width = 80), 
verbose = FALSE, min_term = round(nrow(sim_mat)*0.01), control = list(partition_fun = partition_by_kmeanspp),
column_title = qq("2) @{nrow(sim_mat)} GO terms clustered by 'binary cut'")
), width = 14*2/3, height = 6.5)

#library(cowplot)
pdf("plots/figure8_cvhclust_k2_km2_all_DEGs_primvsnorm_blue.pdf", width = 14, height = 6.5)
print(plot_grid(p1, p2, nrow = 1, rel_widths = c(1, 2)))
dev.off()
p3 <- simplifyGO(sim_mat, ht_list = ht_fdr, word_cloud_grob_param = list(max_width = 80), 
verbose = FALSE, min_term = round(nrow(sim_mat)*0.01), control = list(partition_fun = partition_by_kmeanspp))
write.table(p3, file="simplifyGO_clusters_CVhclust_k2_km2_all_DEGs_primvsnorm.tsv", quote=F, sep="\t")

####
