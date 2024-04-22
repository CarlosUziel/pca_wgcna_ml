###
library('DESeq2')
#library("pheatmap")
library("RColorBrewer")
library("PoiClaClu")
library("genefilter")
#library("heatmap3")
#library("gplots")
library("apeglm")
library("ashr")
library(ggbiplot)
library("data.table")
library("UsingR") 
library("ggplot2")
library("SummarizedExperiment") 
library(circlize)
library("ComplexHeatmap")
library("DT")
library(devtools)
library("BiocParallel")
#register(MulticoreParam(12))
#sessionInfo()

##############################Nov22
#################################Final pots
################################Figure01
##############################pie chart  
df2 <- data.frame(Database=rep(c('DBGAP', 'TCGA', 'TCGA')),
                 Types=rep(c('met (57)', 'prim (500)', 'norm (52)')),
                 Samples=c(57, 500, 52),
                 Labels=c("9,5%", "82%", "8,5%"))
##
#pdf("plots/Figure01D_samples.pdf")
pdf("plots/Figure01A_samples.pdf")
ggplot(df2, aes(x = Database, y = Samples, fill = Types, color = Database)) +
  geom_col(size=.9) +
  geom_text(aes(label = Labels), color = c(1, "white", "white"),
  position = position_stack(vjust = 0.6),show.legend = FALSE) +
  scale_color_manual(values=c("red", "blue")) +
  scale_fill_manual(values=c("#8B3A3A", "#9ACD32","#4A708B")) +
 # scale_fill_viridis_d() +
  coord_polar("y")
  dev.off()

###################################barchart January 2023
###############bar chart
#create data frame
df <- data.frame(Groups=rep(c('Metastatic', 'Metastatic', 'Metastatic', 'Metastatic')),
                 Sites=rep(c('Lymph node', 'Bone', 'Liver', 'Soft tissue')),
                 Samples=c(28, 13, 8, 6))

library(RColorBrewer)
coul <- brewer.pal(4, "Set3") 

#pdf("plots/Figure01C_samples_barplot_04.pdf", width=6, height=5)
pdf("plots/Figure01B_samples_barplot_04.pdf", width=6, height=5)
par(mar = c(4.1, 4.1, 4.1, 10.5))
barplot(as.matrix(df$Samples), horiz=T,
	xlab = "Number of metastatic samples by site",
        col = coul,
        legend.text = c("Lymph node (n=28)", "Bone (n=13)", "Liver (n=8)", "Soft tissue (n=6)"), ylim=c(0,10),
        args.legend = list(title = "Sites", x = "right", inset = c(-0.20, 0))
        #names.arg=c("Lymph node (28)", "Bone (13)", "Liver (8)", "Soft tissue (4)", "Prostate (2)"), 
        #las=2 
        )
dev.off()

##################Figure02A
getwd()
[1] "/media/raheleh/LENOVO/lbi/pcaMet/ML/rna_met/TCGA_PRAD_SU2C_RNASeq_main/rserver/sample_cluster_no_replicate/nov22/deseq2"

dds <- readRDS('sample_cluster_no_replicates_MET_BB+NORM+PRIM__dds.RDS')

dds
class: DESeqDataSet 
dim: 33010 609 

#write.table(colData(dds), file="colData_dds.tsv", quote=F, sep="\t")
###corect names
col2 <- read.table("colData_dds.tsv", header=T, sep="\t", check.names=F, row.names=1)

colData(dds)$sample_type <- as.factor(col2$sample_type)

#colData(dds)$sample_type[colData(dds)$sample_type == PRIM] <- prim

vsd <- vst(dds, blind=FALSE)


################################
#PCAplots
################################
p=plotPCA(vsd, intgroup = c("sample_type"), ntop= Inf,returnData = TRUE)
percentVar <- round(100 * attr(p, "percentVar"))
theme<-theme(panel.background = element_blank(),panel.border=element_rect(fill=NA),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),strip.background=element_blank(),text = element_text(size=15),axis.text.x=element_text(colour="black", size = (12)),axis.text.y=element_text(colour="black", size = (12)),axis.ticks=element_line(colour="black"),plot.margin=unit(c(1,1,1,1),"line"), plot.title = element_text(face = "bold", size = (15)))
d<-ggplot(p,aes(x=PC1,y=PC2,color=(sample_type)))
#d <- ggplot(p, aes(x = condition, y = count, color = condition)) 
#d<-d+geom_point(size = 3)+ 
d<-d+geom_point(alpha = 0.8,size = 4)+ 
xlab(paste0("PC1: ",percentVar[1],"% variance")) +
  ylab(paste0("PC2: ",percentVar[2],"% variance"))
d <- d + theme + scale_color_manual(values=c("#4A708B", "#8B3A3A","#9ACD32"))
#d <- d + labs(y = "Normalized count", x = "SAMPLE TYPE")
d <- d + ggtitle("Global PCA plot with Normalized counts (VST)")
d <- d + theme( axis.line = element_line(colour = "black", size = 1, linetype = "solid"))
d

#pdf("plots/PCAplot_VSD_609samples_sample_cluster_no_replicates_04.pdf")
pdf("plots/Figure02A_PCAplot_VSD_609samples_sample_cluster_no_replicates_04.pdf")
d
dev.off()

############################
#To plot PC3 and PC4 #AdditionalFile_03
############################
library('PCAtools')

p <- pca(assay(vsd), metadata = colData(vsd), removeVar = 0.1)

 #pdf("plots/pairsplot2_counts.VSD_609samples_sample_cluster_no_replicates1_2812.pdf", width=22, height=12)
 pdf("plots/AdditionalFile_03_pairsplot2_counts.VSD_609samples_sample_cluster_no_replicates1_2812.pdf", width=22, height=12)
  pairsplot(p,
    components = getComponents(p, c(1,2,3,4)),
    triangle = TRUE,
    hline = 0, vline = 0,
    pointSize = 0.8,
    gridlines.major = FALSE, gridlines.minor = FALSE,
    colby = 'sample_type', , colkey = c("prim"="#4A708B", "met"="#8B3A3A", "norm"="#9ACD32"),
         legendPosition = 'right', legendLabSize = 12, legendIconSize = 3,
    title = 'Pairs PCA plot', titleLabSize = 22,
    axisLabSize = 14, plotaxes = TRUE,
    margingaps = unit(c(0.1, 0.1, 0.1, 0.1), 'cm'))
  dev.off()
   


###################################Figure02B
#heatmap unsupervised
###################################
topVarGenes <- head(order(rowVars(assay(vsd)), decreasing = TRUE), 500)

mat  <- assay(vsd)[ topVarGenes, ]
mat<-mat-rowMeans(mat)

ha_column = HeatmapAnnotation(df = data.frame(sample_type=(colData(dds)$sample_type)),col = list(sample_type =c("prim"="#4A708B", "met"="#8B3A3A", "norm"="#9ACD32")))

#B9D3EE
#ha_column = HeatmapAnnotation(df = data.frame(Sample_Groups=(colData(dds)$Sample_Groups)),col = list(Sample_Groups =c("Primary"="#7D26CD", "Metastatic"="#8B3A3A", "Normal"="#9ACD32")))

################Remove rows with 0 on prim and norm conditions
write.table(mat, file="plots/vsd.topvar_500.tsv", quote=F, sep="\t")
nrow(mat)
#[1] 500

rownames.remove <- c("ENSG00000197976", "ENSG00000178605", "ENSG00000214717", "ENSG00000182378", "ENSG00000167393", "ENSG00000124333", "ENSG00000169084", "ENSG00000169093", "ENSG00000002586","ENSG00000169100", "ENSG00000100227", "ENSG00000101158", "ENSG00000284976", "ENSG00000181929", "ENSG00000160818")

mat2 <- mat[!(rownames(mat) %in% rownames.remove), ]

nrow(mat2)

#pdf("plots/heatmap_sample_cluster_no_replicates_Normal+Primary_Metastatic_BB_vsd_unsupervised_top500gene_012_final03.pdf")
pdf("plots/Figure02B_heatmap_sample_cluster_no_replicates_Normal+Primary_Metastatic_BB_vsd_unsupervised_top500gene_012_final03.pdf")
ht=Heatmap(mat2, name="VST",
            cluster_columns=F, 
            show_row_dend = T, 
            show_row_names = F, 
            show_column_names = FALSE,
            column_split=dds$sample_type,
            cluster_column_slices = T,
            top_annotation = ha_column,
            column_title = "most variable genes (top 500)",
            #cluster_columns = F,
            cluster_rows = T, 
            heatmap_legend_param = list(title_position = "topcenter", color_bar = "continuous", legend_height = unit(5, "cm"), legend_direction = "horizontal"))
draw(ht, heatmap_legend_side = "top")
dev.off()

###########################################################Figure 03
#venn diagramm 
##########################################################
#library(ggvenn)
library("ggVennDiagram")

resSig <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(resSig)
#[1] 4532

res2Sig <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(res2Sig)
#[1] 1670

#######################################
#Up regulated relative to Mt
#######################################
thr=0.05
category.names=c("PRIM/NORM","MET_BB/PRIM")

resSig.1 <- rownames(subset(subset(resSig, padj<thr),log2FoldChange>0))
resSig.2 <- rownames(subset(subset(res2Sig, padj<thr),log2FoldChange>0))
     
x = list(resSig.2,
	resSig.1)
	
pdf("plots/Venndiagram_Upreg_Condition_Mtastatic_vs_Primary_Normal_padj005_lfc01.pdf")
pdf("plots/Figure03A1_Venndiagram_Upreg_Condition_Mtastatic_vs_Primary_Normal_padj005_lfc01.pdf")
p <- ggVennDiagram(x, label_alpha = 0, color = "black", lwd = 0.8, lty = 1, category.names = c("prim/norm","met/prim"),label_color = "black", label_size = 10)
 # ggplot2::scale_fill_gradient(low="yellow",high = "green")
 p + scale_fill_distiller(palette = "Reds", direction = 1)
dev.off()

#######################################
#Down regulated relative to Mt
#######################################
resSig <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_down_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(resSig)
#[1] 1698

res2Sig <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_down_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(res2Sig)
#[1] 1884

thr=0.05
category.names=c("MET_BB/PRIM", "PRIM/NORM")


resSig.1 <- rownames(subset(subset(resSig, padj<thr),log2FoldChange<0))
resSig.2 <- rownames(subset(subset(res2Sig, padj<thr),log2FoldChange<0))
     
x = list(resSig.2,
	resSig.1)

#pdf("plots/Venndiagram_Downreg_Condition_Mtastatic_vs_Primary_Normal_padj005_lfc01.pdf")	
pdf("plots/Figure03A2_Venndiagram_Downreg_Condition_Mtastatic_vs_Primary_Normal_padj005_lfc01.pdf")
p <- ggVennDiagram(
  x, label_alpha = 0, color = "black", lwd = 0.8, lty = 1,
  category.names = c("prim/norm","met/prim"),
   label_color = "black", label_size = 10) 
 # ggplot2::scale_fill_gradient(low="yellow",high = "green")
 p + scale_fill_distiller(palette = "Blues", direction = 3)
dev.off()

########################################################
#UnSupervised heatmap and GO clusters
########################################################

#############################
##unsupervised analysis
#############################

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

##################Consensus analysis on up and down regulated separately
up1 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(up1)
#[1] 4532

up2 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(up2)
#[1] 1670

down1 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_down_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(down1)
#[1] 1698

down2 <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_down_1_0_deseq_results_unique_02.csv", header=T, sep=(","), row.names=1)
nrow(down2)
#[1] 1884

all1 <- rbind(up1, down1)
nrow(all1)
#[1] 6230

all2 <- rbind(up2, down2)
nrow(all2)
#[1] 3554

#############only for DEGS from prim vs norm
dds2 <- dds[,(colData(dds)$sample_type %in% c("prim","norm"))]

nrow(all2)
#[1] 3554

select_genes<-rownames(subset(all2))
ddsresSig<-(dds2)[select_genes,]
ddsresSig
#dim: 3554 552 

mat1 <- (assay(ddsresSig))
nrow(mat1)
#[1] 3554

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
#
mat = mat_subset[, names(subtype3)]
mat = adjust_matrix(mat)
#1 rows have been removed with zero variance.
#180 rows have been removed with too low variance (sd <= 0.05 quantile)

nrow(mat)
#[1] 3373

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
#pdf("plots/figure8_cvhclust_k2_km2_all_DEGs_primvsnorm_blue.pdf", width = 14, height = 6.5)
pdf("plots/Figure03B_figure8_cvhclust_k2_km2_all_DEGs_primvsnorm_blue.pdf", width = 14, height = 6.5)
print(plot_grid(p1, p2, nrow = 1, rel_widths = c(1, 2)))
dev.off()
p3 <- simplifyGO(sim_mat, ht_list = ht_fdr, word_cloud_grob_param = list(max_width = 80), 
verbose = FALSE, min_term = round(nrow(sim_mat)*0.01), control = list(partition_fun = partition_by_kmeanspp))
write.table(p3, file="simplifyGO_clusters_CVhclust_k2_km2_all_DEGs_primvsnorm.tsv", quote=F, sep="\t")

####


#############only for DEGS from met vs prim
dds3 <- dds[,(colData(dds)$sample_type %in% c("met","prim"))]
dds3
nrow(all1)
#[1] 6230

select_genes<-rownames(subset(all1))
ddsresSig<-(dds3)[select_genes,]
ddsresSig
#dim: 6230 557 

mat1 <- (assay(ddsresSig))
nrow(mat1)
#[1] 6230

mat_subset <- subset(mat1, !is.na(rownames(mat1)))
#head(mat_subset)
nrow(mat_subset)

subtype = (colData(dds3)) 
head(subtype)

subtype = (colData(dds3)[, 2, drop=F])
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
#
mat = mat_subset[, names(subtype3)]
mat = adjust_matrix(mat)
#1 rows have been removed with zero variance.
#312 rows have been removed with too low variance (sd <= 0.05 quantile)

nrow(mat)
#[1] 5917

rlall1 = run_all_consensus_partition_methods(
    mat,
  #  top_value_method = c("SD", "MAD", 'CV', 'ATC'),
  #  partition_method = c("hclust", "kmeans", "skmeans", "pam", "mclust),
    mc.cores = 4,
    anno = data.frame(subtype = subtype3),
    anno_col = list(subtype = subtype_col)
)
saveRDS(rlall1, file = "resSig_met_vs_Prim_all_DEGs_dds_cleaned_PRAD_DbGAP_03.rds")

rlall1 = readRDS("resSig_met_vs_Prim_all_DEGs_dds_cleaned_PRAD_DbGAP_03.rds")

################
res = rlall1["CV:hclust"]
set.seed(123)
#res = golub_cola["ATC:skmeans"]
df = get_signatures(res, k = 2, row_km = 2, plot = FALSE)
write.table(df, file="result_top_signatures_CVhclust_k2_km2_all_DEGs_metvsprim.tsv", quote=F, sep="\t")
cl = get_classes(res, k = 2)[, 1]
cl[cl == 1] = 4; cl[cl == 2] = 1; cl[cl == 4] = 2
m = get_matrix(res)[df$which_row, ]
m = t(scale(t(m)))
nrow(m)
#[1] 5367

write.table(m, file="top_signatures_CVhclust_k2_km2_all_DEGs_metvsprim.tsv", quote=F, sep="\t")

ht = Heatmap(m, name = "Scaled\nexpression",
col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
top_annotation = HeatmapAnnotation(
sample_class = cl,
sample_type = get_anno(res)[, 1],
col = list(sample_class = cola:::brewer_pal_set2_col[1:3],
sample_type = c("prim"="#4A708B", "met"="#8B3A3A"))),
#sample_type = c("prim"="#4A708B", "norm"="#9ACD32"))),
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
write.table(pm, file="GO_clusters_CVhclust_k2_km2_all_DEGs_metvsprim.tsv", quote=F, sep="\t")

names(lt)
head(lt[[1]][, 1:7])
write.table((lt[[1]][, 1:7]), file="BP_km1_clusters_CVhclust_k2_km2_all_DEGs_metvsprim.tsv", quote=F, sep="\t")
head(lt[[2]][, 1:7])
write.table((lt[[2]][, 1:7]), file="BP_km2_clusters_CVhclust_k2_km2_all_DEGs_metvsprim.tsv", quote=F, sep="\t")

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
#pdf("plots/figure8_cvhclust_k2_km2_all_DEGs_metvsprim_blue.pdf", width = 14, height = 6.5)
pdf("plots/Figure03C_figure8_cvhclust_k2_km2_all_DEGs_metvsprim_blue.pdf", width = 14, height = 6.5)
print(plot_grid(p1, p2, nrow = 1, rel_widths = c(1, 2)))
dev.off()
p3 <- simplifyGO(sim_mat, ht_list = ht_fdr, word_cloud_grob_param = list(max_width = 80), 
verbose = FALSE, min_term = round(nrow(sim_mat)*0.01), control = list(partition_fun = partition_by_kmeanspp))
write.table(p3, file="simplifyGO_clusters_CVhclust_k2_km2_all_DEGs_metvsprim.tsv", quote=F, sep="\t")

################Figure04
#############Figure for top CSPR
#####
# Libraries
library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(viridis)

getwd()
#[1] "/media/raheleh/LENOVO/lbi/pcaMet/ML/rna_met/TCGA_PRAD_SU2C_RNASeq_main/rserver/sample_cluster_no_replicate/nov22/deseq2"

data <- read.csv("CSPR_top_Metastatic_BB_vs_Primary_609samples_up_final.csv", header=T, sep=",")

#
#ggsave("plots/CSPR_top_Metastatic_BB_vs_Primary_609samples_up_062.pdf", width = 12, height = 10)
ggplot(data, aes(x=reorder(Description, log2FoldChange), y=log2FoldChange, fill=X.Log10.Padj.)) + geom_bar(stat="identity", position="identity",width=0.9,color = 'black', size=0.8) +coord_flip()+   
xlab("Top Up-regulated cell surface receptors") +
ylab("Log2FoldChange")+
scale_color_viridis(option="F") +
#theme_bw() +
theme_ipsum() + 
theme( plot.background = element_blank()
   ,panel.grid.major = element_blank()
   ,panel.grid.minor = element_blank()
   ,panel.border = element_rect(colour = "black", fill=NA, size=1)) 
    #theme(axis.line = element_line(color = 'black')+ theme(aspect.ratio=1))+ labs(x=Top Up-regulated Cell Surface Receptors", y=c("-Log10(P-value)"),fill = "Padj")
ggsave("plots/Figure04A_CSPR_top_Metastatic_BB_vs_Primary_609samples_up_bar.png", device = "png", dpi=1600)
#ggsave("plots/CSPR_top_Metastatic_BB_vs_Primary_609samples_up_bar.png", device = "png", dpi=1600)


############################################Dec 2022
#PCA plot for all 776 samples #AdditionalFile01

getwd()
#[1] "/media/raheleh/LENOVO/lbi/pcaMet/ML/rna_met/TCGA_PRAD_SU2C_RNASeq_main/rserver/sample_cluster_no_replicate/nov22/deseq2"

#dds776 <- load('deseq2_dds_metastatic_primary_normal_776samples_sample_cluster_noreplicate.RDS')
dds776 <- load('deseq2_dds_metastaticBB_primary_normal_776samples_sample_cluster_no_replicate.RData')
dds776
#[1] "dds"
 dds
#class: DESeqDataSet 
#dim: 55794 776 

head(colData(dds))

colnames(colData(dds))
#[1] "patient_id"                   "sample_type"                 
#[3] "file_name"                    "release_year"                
#[5] "sample_cluster"               "sample_cluster_no_replicates"
#[7] "sizeFactor"                   "replaceable"                 

colnames(colData(dds)) <- c("patient_id", "type","file_name","release_year", "Sample_Cluster", "Sample_Groups", "sizeFactor", "replaceable")

colData(dds)$sample_type <- as.factor(dds$Sample_Groups)

#colData(dds)$sample_type[colData(dds)$sample_type == PRIM] <- prim

#vsd <- vst(dds, blind=FALSE)
vsd776 <- load('deseq2_vsd_metastatic_primary_normal_776samples_sample_cluster_no_replicate.RData')

vsd776
#[1] "vsd"
 vsd
#class: DESeqTransform 
#dim: 55794 776 

colnames(colData(vsd)) <- c("patient_id", "type","file_name","release_year", "Sample_Cluster", "Sample_Groups", "sizeFactor", "replaceable")

colData(vsd)$sample_type <- as.factor(vsd$Sample_Groups)

vsd5 <- vsd[,(colData(vsd)$sample_type %in% c("Primary", "Metastatic_BB", "Metastatic_AA", "Normal", "Endocrine"))]

#7 Levels: "Primary", "Metastatic_BB", "Metastatic_AA", "Normal", "Endocrine"

################################
#PCAplots
################################

p=plotPCA(vsd5, intgroup = c("sample_type"), ntop= Inf,returnData = TRUE)
percentVar <- round(100 * attr(p, "percentVar"))
theme<-theme(panel.background = element_blank(),panel.border=element_rect(fill=NA),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),strip.background=element_blank(),text = element_text(size=15),axis.text.x=element_text(colour="black", size = (12)),axis.text.y=element_text(colour="black", size = (12)),axis.ticks=element_line(colour="black"),plot.margin=unit(c(1,1,1,1),"line"), plot.title = element_text(face = "bold", size = (15)))
d<-ggplot(p,aes(x=PC1,y=PC2,color=(sample_type)))
#d <- ggplot(p, aes(x = condition, y = count, color = condition)) 
#d<-d+geom_point(size = 3)+ 
d<-d+geom_point(alpha = 0.8,size = 4)+ 
xlab(paste0("PC1: ",percentVar[1],"% variance")) +
  ylab(paste0("PC2: ",percentVar[2],"% variance"))
d <- d + theme + scale_color_manual(values=c("#4A708B", "#8B3A3A","pink","#9ACD32", "purple"))
#d <- d + labs(y = "Normalized count", x = "SAMPLE TYPE")
d <- d + ggtitle("Global PCA plot with Normalized counts (VST)")
d <- d + theme( axis.line = element_line(colour = "black", size = 1, linetype = "solid"))
d

#pdf("plots/PCAplot_VSD_776samples_sample_cluster_no_replicates_05.pdf")
pdf("plots/AdditionalFile01_PCAplot_VSD_776samples_sample_cluster_no_replicates_05.pdf")
d
dev.off()
   
###########################Figure04B
#Active Pathways
########################
library(ActivePathways)
setwd("june22_609samples/active_pathways")
#####################ActivePathway analysis with DEGs Met vs Pri up
#library(ActivePathways)
scores <- read.table("matrix_sample_cluster_no_replicates_Metastatic_BB+Normal+Primary_Metastatic_BB_vs_Primary_padj_0_05_up_1_0_deseq_results_2.csv", header = TRUE, sep="\t")

scores <- scores[!duplicated(scores$Gene), ]

write.table(scores, file="matrix_sample_cluster_no_replicates_Metastatic_BB+Normal+Primary_Metastatic_BB_vs_Primary_padj_0_05_up_1_0_deseq_results_3.csv",header = TRUE, sep="\t")

scores <- read.table("matrix_sample_cluster_no_replicates_Metastatic_BB+Normal+Primary_Metastatic_BB_vs_Primary_padj_0_05_up_1_0_deseq_results_3.csv", header = TRUE, row.names = 'Gene', sep="\t")

#rownames(scores)<- scores$Gene
#scores2 <- subset(scores, select=-(Gene))

scores <- as.matrix(scores)
scores[is.na(scores)] <- 1

####
gmt.GOBP <- ('gprofiler_gmt/hsapiens.GOBP.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.GOBP, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_GOBP_Sig05.csv")

res <- ActivePathways(scores, gmt.GOBP, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_GOBP_Sig05_")

##
gmt.GOCC <- ('gprofiler_gmt/hsapiens.GOCC.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.GOCC, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_GOCC_Sig05.csv")

res <- ActivePathways(scores, gmt.GOCC, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_GOCC_Sig05_")

##
gmt.GOMF <- ('gprofiler_gmt/hsapiens.GOMF.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.GOMF, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_GOMF_Sig05.csv")

res <- ActivePathways(scores, gmt.GOMF, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_GOMF_Sig05_")
###
gmt.HPA <- ('gprofiler_gmt/hsapiens.HPA.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.HPA, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_HPA_Sig05.csv")

res <- ActivePathways(scores, gmt.HPA, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_HPA_Sig05_")

###
gmt.HP <- ('gprofiler_gmt/hsapiens.HP.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.HP, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_HP_Sig05.csv")

res <- ActivePathways(scores, gmt.HP, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_HP_Sig05_")

###
gmt.WP <- ('gprofiler_gmt/hsapiens.WP.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.WP, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_WP_Sig05.csv")

res <- ActivePathways(scores, gmt.WP, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_WP_Sig05_")
######
gmt.CORUM <- ('gprofiler_gmt/hsapiens.CORUM.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.CORUM, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_CORUM_Sig05.csv")

res <- ActivePathways(scores, gmt.CORUM, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_CORUM_Sig05_")

###
gmt.MIRNA <- ('gprofiler_gmt/hsapiens.MIRNA.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.MIRNA, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_MIRNA_Sig05.csv")

res <- ActivePathways(scores, gmt.MIRNA, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_MIRNA_Sig05_")

###
gmt.full <- ('gprofiler_gmt/gprofiler.full.hsapiens.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.full, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_full_Sig05.csv")

res <- ActivePathways(scores, gmt.full, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_full_Sig05_")

###
gmt.REAC <- ('gprofiler_gmt/hsapiens.REAC.name.gmt')

enriched_pathways <- ActivePathways(scores, gmt.REAC, significant=0.05) 

export_as_CSV(enriched_pathways, "enriched_pathways_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_REAC_Sig05.csv")

res <- ActivePathways(scores, gmt.REAC, significant=0.05, cytoscape.file.tag = "enrichmentMap_DEGs_Metastatic_BB_vs_Primary_padj_0_05_up_1_REAC_Sig05_")

####################





