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
#sessionInfo()
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

#######April22

getwd()
[1] "/share/computing/raheleh/projects/storage/TCGA_PRAD_SU2C_RNASeq/rserver"

#import the data

directory="/share/computing/raheleh/projects/storage/TCGA_PRAD_SU2C_RNASeq/rserver/star_counts"

sampleFiles <- grep("*.tsv",list.files(directory),value=TRUE)
sampleFiles

coldata <- read.csv("/share/computing/raheleh/projects/storage/TCGA_PRAD_SU2C_RNASeq/rserver/samples_annotation_tcga_prad_su2c_clusters_metBB_pri_nr.csv", row.names=1, header=T, check.names=FALSE)

sampleTable<-data.frame(sampleName=sampleFiles, fileName=sampleFiles, condition=coldata)
#sampleTable
colnames(sampleTable) <- c("sampleName", "fileName","patient_id", "Run", "Sample_Type", "file_name", "release_year", "sample_cluster", "sample_cluster_no_replicates", "Condition")
sampleTable$sample_cluster_no_replicates <- factor(sampleTable$sample_cluster_no_replicates)
sampleTable$sample_cluster <- factor(sampleTable$sample_cluster)
sampleTable$Condition <- factor(sampleTable$Condition)
sampleTable$Run <- factor(sampleTable$Run)
sampleTable$Sample_Type <- factor(sampleTable$Sample_Type)
sampleTable$release_year <- factor(sampleTable$release_year)
#sampleTable$Patient <- factor(sampleTable$Patient)

ddsHTSeq<-DESeqDataSetFromHTSeqCount(sampleTable=sampleTable, directory=directory, design=~ Sample_Type)

save(ddsHTSeq,file="sample_cluster_no_replicate/deseq2_ddsHTSeq_metastaticBB_primary_normal_609samples_sample_cluster_noreplicate_noEndocrine.RData")
#save(ddsHTSeq,file="sample_cluster/deseq2_ddsHTSeq_metastatic_primary_normal_762samples_sample_cluster_noEndocrine.RData")

keep <- rowSums(counts(ddsHTSeq)) >= 10
ddsHTSeq <- ddsHTSeq[keep,]

colData(ddsHTSeq)$Sample_Type<-factor(colData(ddsHTSeq)$Sample_Type, levels=c("Primary","Metastatic",
"Normal"))

dds <- DESeq(ddsHTSeq)

save(dds,file="sample_cluster_no_replicate/deseq2_dds_metastaticBB_primary_normal_609samples_sample_cluster.RData")
write.csv(as.data.frame(assay(dds)), file="sample_cluster_no_replicate/deseq2_dds_metastaticBB_primary_normal_609samples_sample_cluster_no_replicate.csv")

vsd <- vst(dds, blind=FALSE)
save(vsd,file="sample_cluster_no_replicate/deseq2_vsd_metastaticBB_primary_normal_609samples_sample_cluster.RData")
write.csv(as.data.frame(assay(vsd)), file="sample_cluster_no_replicate/deseq2_vsd_metastaticBB_primary_normal_609samples_sample_cluster_no_replicate.csv")

mat <- assay(vsd)
mat2 <- limma::removeBatchEffect(mat, vsd$release_year)
#assay(vsd) <- mat
#counts_batch_corrected <- assay(vsd)


#############################
#collecting Results Condition_Metastatic_vs_Primary
#############################
resultsNames(dds) # lists the coefficients
#[1] "Intercept"                         "Sample_Type_Metastatic_vs_Primary"
#[3] "Sample_Type_Normal_vs_Primary"    

#res <- results(dds, name="Sample_Type_Metastatic_vs_Primary")
# or to shrink log fold changes association with condition:
#res <- results(dds, contrast=c("condition","Metastatic_BB","Primary"))
#result = deseq_results(deseq, contrast=ro.StrVector(contrast), alpha=0.99999)
#results[(test, control)] = lfc_shrink(dds=deseq, res=result, **{"type": "ashr"})
#res <- lfcShrink(dds, coef="Sample_Type_Metastatic_vs_Primary", type="apeglm", lfcThreshold=1)
#nrow(res)

#######################ashr
res <- lfcShrink(dds, coef="Sample_Type_Metastatic_vs_Primary", type="ashr")
#res <- lfcShrink(dds, contrast=c("Sample_Type","Metastatic","Primary"), type="ashr")

nrow(res)
#[1] 53954

#sum(resMTvsNR$padj < 0.01, na.rm=TRUE)
#resMTvsNR.05 <- results(dds, alpha=0.05)
#summary(resMTvsNR.05)

#library("AnnotationDbi")
#library("org.Hs.eg.db")
#library("rnaseqGene")


res$SYMBOL <- mapIds(org.Hs.eg.db,
                     keys=row.names(res),
                     column="SYMBOL",
                     keytype="ENSEMBL",
                     multiVals="first")
write.csv(as.data.frame(res), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_all.csv")

###Extracting the results adjusted p-adj < 0.05
resSig <- subset(res, padj < 0.05 & abs(log2FoldChange)>1)
 nrow(resSig)
#[1]  13089

write.csv(as.data.frame(resSig), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc01.csv")

up <- subset(resSig, log2FoldChange > 0)
write.table(up[order(up$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc01_upreg.csv")
 nrow(up)
#[1] 9922

down <- subset(resSig, log2FoldChange < 0)
write.table(down[order(down$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc01_downreg.csv")
 nrow(down)
#[1] 3167

###Extracting the results adjusted pvalue < 0.05
resSig1 <- subset(res, pvalue < 0.05 & abs(log2FoldChange)>1)
 nrow(resSig1)
#[1]  13135

write.csv(as.data.frame(resSig1), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc01.csv")

up1 <- subset(resSig1, log2FoldChange > 0)
write.table(up1[order(up1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc01_upreg.csv")
 nrow(up1)
#[1] 9966

down1 <- subset(resSig1, log2FoldChange < 0)
write.table(down1[order(down1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc01_downreg.csv")
 nrow(down1)
#[1] 3169

#######################
###Extracting the results adjusted p-adj < 0.05
resSig <- subset(res, padj < 0.05 & abs(log2FoldChange)>1.5)
 nrow(resSig)
#[1]  7209

write.csv(as.data.frame(resSig), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc1.5.csv")

up <- subset(resSig, log2FoldChange > 0)
write.table(up[order(up$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc1.5_upreg.csv")
 nrow(up)
#[1] 5457

down <- subset(resSig, log2FoldChange < 0)
write.table(down[order(down$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc1.5_downreg.csv")
 nrow(down)
#[1] 1752

###Extracting the results adjusted pvalue < 0.05
resSig1 <- subset(res, pvalue < 0.05 & abs(log2FoldChange)>1.5)
 nrow(resSig1)
#[1]  7209

write.csv(as.data.frame(resSig1), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc1.5.csv")

up1 <- subset(resSig1, log2FoldChange > 0)
write.table(up1[order(up1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc1.5_upreg.csv")
 nrow(up1)
#[1] 5457

down1 <- subset(resSig1, log2FoldChange < 0)
write.table(down1[order(down1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc1.5_downreg.csv")
 nrow(down1)
#[1] 1752

###Extracting the results adjusted p-adj < 0.05
resSig <- subset(res, padj < 0.05 & abs(log2FoldChange)>2)
 nrow(resSig)
#[1]  3724

write.csv(as.data.frame(resSig), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc2.csv")

up <- subset(resSig, log2FoldChange > 0)
write.table(up[order(up$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc2_upreg.csv")
 nrow(up)
#[1] 2748

down <- subset(resSig, log2FoldChange < 0)
write.table(down[order(down$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_padj005_lfc2_downreg.csv")
 nrow(down)
#[1] 976

###Extracting the results adjusted pvalue < 0.05
resSig1 <- subset(res, pvalue < 0.05 & abs(log2FoldChange)>2)
 nrow(resSig1)
#[1]  3724

write.csv(as.data.frame(resSig1), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc2.csv")

up1 <- subset(resSig1, log2FoldChange > 0)
write.table(up1[order(up1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc2_upreg.csv")
 nrow(up1)
#[1] 2748

down1 <- subset(resSig1, log2FoldChange < 0)
write.table(down1[order(down1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_ashr_pvalue005_lfc2_downreg.csv")
 nrow(down1)
#[1] 976

#############################apeglm
res <- lfcShrink(dds, coef="Sample_Type_Metastatic_vs_Primary", type="apeglm")


nrow(res)
#[1] 53954

#sum(resMTvsNR$padj < 0.01, na.rm=TRUE)
#resMTvsNR.05 <- results(dds, alpha=0.05)
#summary(resMTvsNR.05)

library("AnnotationDbi")
library("org.Hs.eg.db")
library("rnaseqGene")


res$SYMBOL <- mapIds(org.Hs.eg.db,
                     keys=row.names(res),
                     column="SYMBOL",
                     keytype="ENSEMBL",
                     multiVals="first")
write.csv(as.data.frame(res), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_apeglm_all.csv")

###Extracting the results adjusted p-adj < 0.05
resSig <- subset(res, padj < 0.05 & abs(log2FoldChange)>1)
 nrow(resSig)
#[1]  13159

write.csv(as.data.frame(resSig), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_apeglm_padj005_lfc01.csv")

up <- subset(resSig, log2FoldChange > 0)
write.table(up[order(up$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_apeglm_padj005_lfc01_upreg.csv")
 nrow(up)
#[1] 9949

down <- subset(resSig, log2FoldChange < 0)
write.table(down[order(down$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_apeglm_padj005_lfc01_downreg.csv")
 nrow(down)
#[1] 3210

###Extracting the results adjusted pvalue < 0.05
resSig1 <- subset(res, pvalue < 0.05 & abs(log2FoldChange)>1)
 nrow(resSig1)
#[1]  13388

write.csv(as.data.frame(resSig1), file="sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_apeglm_pvalue005_lfc01.csv")

up1 <- subset(resSig1, log2FoldChange > 0)
write.table(up1[order(up1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_apeglm_pvalue005_lfc01_upreg.csv")
 nrow(up1)
#[1] 10117

down1 <- subset(resSig1, log2FoldChange < 0)
write.table(down1[order(down1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Metastatic_BB_vs_Primary_lfcshrink_apeglm_pvalue005_lfc01_downreg.csv")
 nrow(down1)
#[1] 3271

#############################
#collecting Results Condition_Primary_vs_Normal
#############################
resultsNames(dds) # lists the coefficients
#[1] "Intercept"                         "Sample_Type_Metastatic_vs_Primary"
#[3] "Sample_Type_Normal_vs_Primary"    

#res <- results(dds, name="Sample_Type_Metastatic_vs_Primary")
# or to shrink log fold changes association with condition:
#res <- results(dds, contrast=c("condition","Metastatic_BB","Primary"))
#result = deseq_results(deseq, contrast=ro.StrVector(contrast), alpha=0.99999)
#results[(test, control)] = lfc_shrink(dds=deseq, res=result, **{"type": "ashr"})
#res <- lfcShrink(dds, coef="Sample_Type_Metastatic_vs_Primary", type="apeglm", lfcThreshold=1)
#nrow(res)

#######################ashr
res <- lfcShrink(dds, contrast=c("Sample_Type","Primary","Normal"), type="ashr")
#res <- lfcShrink(dds, coef="Sample_Type_Normal_vs_Primary", type="ashr")
nrow(res)
#[1] 53954

#sum(resMTvsNR$padj < 0.01, na.rm=TRUE)
#resMTvsNR.05 <- results(dds, alpha=0.05)
#summary(resMTvsNR.05)

#library("AnnotationDbi")
#library("org.Hs.eg.db")
#library("rnaseqGene")


res$SYMBOL <- mapIds(org.Hs.eg.db,
                     keys=row.names(res),
                     column="SYMBOL",
                     keytype="ENSEMBL",
                     multiVals="first")
write.csv(as.data.frame(res), file="sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_all.csv")

###Extracting the results adjusted p-adj < 0.05
resSig <- subset(res, padj < 0.05 & abs(log2FoldChange)>1)
 nrow(resSig)
#[1]  398

write.csv(as.data.frame(resSig), file="sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc01.csv")

up <- subset(resSig, log2FoldChange > 0)
write.table(up[order(up$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc01_upreg.csv")
 nrow(up)
#[1] 371

down <- subset(resSig, log2FoldChange < 0)
write.table(down[order(down$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc01_downreg.csv")
 nrow(down)
#[1] 27

###Extracting the results adjusted pvalue < 0.05
resSig1 <- subset(res, pvalue < 0.05 & abs(log2FoldChange)>1)
 nrow(resSig1)
#[1]  398

write.csv(as.data.frame(resSig1), file="sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc01.csv")

up1 <- subset(resSig1, log2FoldChange > 0)
write.table(up1[order(up1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc01_upreg.csv")
 nrow(up1)
#[1] 371

down1 <- subset(resSig1, log2FoldChange < 0)
write.table(down1[order(down1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc01_downreg.csv")
 nrow(down1)
#[1] 27

#######################
###Extracting the results adjusted p-adj < 0.05
resSig <- subset(res, padj < 0.05 & abs(log2FoldChange)>1.5)
 nrow(resSig)
#[1]  189

write.csv(as.data.frame(resSig), file="sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc1.5.csv")

up <- subset(resSig, log2FoldChange > 0)
write.table(up[order(up$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc1.5_upreg.csv")
 nrow(up)
#[1] 173

down <- subset(resSig, log2FoldChange < 0)
write.table(down[order(down$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc1.5_downreg.csv")
 nrow(down)
#[1] 16

###Extracting the results adjusted pvalue < 0.05
resSig1 <- subset(res, pvalue < 0.05 & abs(log2FoldChange)>1.5)
 nrow(resSig1)
#[1]  189

write.csv(as.data.frame(resSig1), file="sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc1.5.csv")

up1 <- subset(resSig1, log2FoldChange > 0)
write.table(up1[order(up1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc1.5_upreg.csv")
 nrow(up1)
#[1] 173

down1 <- subset(resSig1, log2FoldChange < 0)
write.table(down1[order(down1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc1.5_downreg.csv")
 nrow(down1)
#[1] 16

###Extracting the results adjusted p-adj < 0.05
resSig <- subset(res, padj < 0.05 & abs(log2FoldChange)>2)
 nrow(resSig)
#[1]  107

write.csv(as.data.frame(resSig), file="sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc2.csv")

up <- subset(resSig, log2FoldChange > 0)
write.table(up[order(up$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc2_upreg.csv")
 nrow(up)
#[1] 98

down <- subset(resSig, log2FoldChange < 0)
write.table(down[order(down$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_padj005_lfc2_downreg.csv")
 nrow(down)
#[1] 9

###Extracting the results adjusted pvalue < 0.05
resSig1 <- subset(res, pvalue < 0.05 & abs(log2FoldChange)>2)
 nrow(resSig1)
#[1]  107

write.csv(as.data.frame(resSig1), file="sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc2.csv")

up1 <- subset(resSig1, log2FoldChange > 0)
write.table(up1[order(up1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc2_upreg.csv")
 nrow(up1)
#[1] 98

down1 <- subset(resSig1, log2FoldChange < 0)
write.table(down1[order(down1$log2FoldChange, decreasing=TRUE),],"sample_cluster_no_replicate/results_Sample_Type_Primary_vs_Normal_lfcshrink_ashr_pvalue005_lfc2_downreg.csv")
 nrow(down1)
#[1] 9

#######################
#plots
#######################
##
#p=plotPCA(vsd, intgroup = c("sample_cluster_no_replicates", "release_year"), ntop=500)
pdf("sample_cluster/PCAplot_VSD_batch_ntop500_0422.pdf")
p
dev.off()

#p=plotPCA(vsd, intgroup = c("sample_cluster_no_replicates", "release_year"), ntop=100)
pdf("sample_cluster/PCAplot_VSD_batch_ntop100_0422.pdf")
p
dev.off()

#p=plotPCA(vsd, intgroup = c("sample_cluster_no_replicates", "release_year"), ntop=1000)
pdf("sample_cluster/PCAplot_VSD_batch_ntop1000_0422.pdf")
p
dev.off()

p=plotPCA(vsd, intgroup = c("Sample_Type"), ntop= Inf)
pdf("sample_cluster_no_replicate/PCAplot_VSD_609samples_ntopInf_0422.pdf")
p
dev.off()

###
# also possible to perform custom transformation:
     dds2 <- estimateSizeFactors(dds)
     # shifted log of normalized counts
     se <- SummarizedExperiment(log2(counts(dds2, normalized=TRUE) + 1),
                                colData=colData(dds2))
     # the call to DESeqTransform() is needed to
     # trigger our plotPCA method.
    pdf("sample_cluster_no_replicate/PCAplot_logdds_metBB_609samples_0422_Condition.pdf")
    plotPCA( DESeqTransform( se ) , intgroup = c("Sample_Type"), ntop=500)
    dev.off()
    
     pdf("sample_cluster_no_replicate/PCAplot_logdds_metBB_609samples_0422_Condition_releaseyear.pdf")
     plotPCA( DESeqTransform( se ) , intgroup = c("Sample_Type", "release_year"), ntop=500)
     dev.off()
#
library(ggbiplot)
df <- mat2
 tdf=t(df)

pca <- prcomp(tdf, center = TRUE, scale. = FALSE) 

 g <- ggbiplot(pcobj = pca, scale = 1, obs.scale = 1, var.scale = 1, 
                groups = colData(vsd)$Sample_Type, ellipse = TRUE, 
                circle = TRUE, var.axes = FALSE)
                
  g <- g + scale_color_discrete(name = 'xx')
  g <- g + theme(legend.direction = 'horizontal', 
                 legend.position = 'top')
  print(g)
  pdf("sample_cluster_no_replicate/PCAplot_lima_batchcorrection_dbgap_tcga_03.pdf")
 g
 dev.off()

###
df <- mat
 tdf=t(df)

pca <- prcomp(tdf, center = TRUE, scale. = FALSE) 

 g <- ggbiplot(pcobj = pca, scale = 1, obs.scale = 1, var.scale = 1, 
                groups = colData(vsd)$Sample_Type, ellipse = TRUE, 
                circle = TRUE, var.axes = FALSE)
                
  g <- g + scale_color_discrete(name = 'xx')
  g <- g + theme(legend.direction = 'horizontal', 
                 legend.position = 'top')
  print(g)
  pdf("sample_cluster_no_replicate/biPCAplot_vsd_dbgap_tcga_609sample_cluster_no_replicate.pdf")
 g
 dev.off()


###
pcaData <- plotPCA(vsd, intgroup=c("Sample_Type"), ntop= Inf, returnData=TRUE)
percentVar <- round(100 * attr(pcaData, "percentVar"))
pdf("sample_cluster_no_replicate/PCAplot_VSD_609samples_dbgap_tcga_Sample_Type.pdf")
ggplot(pcaData, aes(PC1, PC2, color=Sample_Type, shape=Sample_Type)) +
  geom_point(size=3) +
  xlab(paste0("PC1: ",percentVar[1],"% variance")) +
  ylab(paste0("PC2: ",percentVar[2],"% variance")) + 
  coord_fixed()
 dev.off() 
##


"#006400"=darkgreen, Normal
"#104E8B"=dodgerblue4, Primary
"#6495ED"=cornflowerblue, Primary
"#CD0000"=RED3, Metastatic_BB
"#CD3700"=ORANGERED3, Metastatic_AA
"#8B4789"=orchid4, Metastatic_A
"#551A8B"=purple4, Metastatic_B

#library("ggplot2")
p=plotPCA(vsd, intgroup = c("Sample_Type"), ntop= Inf,returnData = TRUE)
theme<-theme(panel.background = element_blank(),panel.border=element_rect(fill=NA),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),strip.background=element_blank(),text = element_text(size=15),axis.text.x=element_text(colour="black", size = (12)),axis.text.y=element_text(colour="black", size = (12)),axis.ticks=element_line(colour="black"),plot.margin=unit(c(1,1,1,1),"line"), plot.title = element_text(face = "bold", size = (15)))
d<-ggplot(p,aes(x=PC1,y=PC2,color=(Sample_Type),shape=Sample_Type ))
#d <- ggplot(p, aes(x = condition, y = count, color = condition)) 
#d<-d+geom_point(size = 3)+ 
d<-d+geom_point(alpha = 0.8,size = 4)+ 
xlab(paste0("PC1: ",percentVar[1],"% variance")) +
  ylab(paste0("PC2: ",percentVar[2],"% variance"))
d <- d + theme + scale_color_manual(values=c("#6495ED", "#CD0000","#006400"))
#d <- d + labs(y = "Normalized count", x = "SAMPLE TYPE")
d <- d + ggtitle("Global PCA plot Normalized counts")
d <- d + theme( axis.line = element_line(colour = "black", size = 1, linetype = "solid"))
d

pdf("sample_cluster_no_replicate/PCAplot_VSD_dbgap_tcga_609samples_Sample_Type.pdf")
d
dev.off()

####
p=plotPCA(vsd, intgroup = c("Sample_Type"), ntop= Inf,returnData = TRUE)
theme<-theme(panel.background = element_blank(),panel.border=element_rect(fill=NA),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),strip.background=element_blank(),text = element_text(size=15),axis.text.x=element_text(colour="black", size = (12)),axis.text.y=element_text(colour="black", size = (12)),axis.ticks=element_line(colour="black"),plot.margin=unit(c(1,1,1,1),"line"), plot.title = element_text(face = "bold", size = (15)))
d<-ggplot(p,aes(x=PC1,y=PC2,color=(Sample_Type)))
#d <- ggplot(p, aes(x = condition, y = count, color = condition)) 
#d<-d+geom_point(size = 3)+ 
d<-d+geom_point(alpha = 0.8,size = 4)+ 
xlab(paste0("PC1: ",percentVar[1],"% variance")) +
  ylab(paste0("PC2: ",percentVar[2],"% variance"))
d <- d + theme + scale_color_manual(values=c("#6495ED", "#CD0000","#006400"))
#d <- d + labs(y = "Normalized count", x = "SAMPLE TYPE")
d <- d + ggtitle("Global PCA plot Normalized counts")
d <- d + theme( axis.line = element_line(colour = "black", size = 1, linetype = "solid"))
d

pdf("sample_cluster_no_replicate/PCAplot_VSD_dbgap_tcga_609samples_Condition02.pdf")
d
dev.off()


###
#p=plotPCA(vsd, intgroup = c("Condition", "release_year"),returnData = TRUE)
theme<-theme(panel.background = element_blank(),panel.border=element_rect(fill=NA),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),strip.background=element_blank(),text = element_text(size=15),axis.text.x=element_text(colour="black", size = (12)),axis.text.y=element_text(colour="black", size = (12)),axis.ticks=element_line(colour="black"),plot.margin=unit(c(1,1,1,1),"line"), plot.title = element_text(face = "bold", size = (15)))
d<-ggplot(p,aes(x=PC1,y=PC2,color=(Condition),shape=release_year ))
#d <- ggplot(p, aes(x = condition, y = count, color = condition)) 
#d<-d+geom_point(size = 3)+ 
d<-d+geom_point(alpha = 0.8,size = 4)+ 
xlab(paste0("PC1: ",percentVar[1],"% variance")) +
  ylab(paste0("PC2: ",percentVar[2],"% variance"))
d <- d + theme + scale_color_manual(values=c("#6495ED","#CD0000","#CD3700","#006400","#8B4789", "#551A8B"))
#d <- d + labs(y = "Normalized count", x = "SAMPLE TYPE")
d <- d + ggtitle("Global PCA plot Normalized counts")
d <- d + theme( axis.line = element_line(colour = "black", size = 1, linetype = "solid"))
d

pdf("sample_cluster_no_replicate/PCAplot_VSD_dbgap_tcga_762_Condition_releaseyear.pdf")
d
dev.off()

############################
library(scatterplot3d)

df <- assay(vsd)

tdf=t(df)

tpca <- prcomp(tdf, scale.=TRUE)

main=as.factor(colData(vsd)$Sample_Type)
main=as.data.frame(main)

tpcagr=cbind(tpca$x,main)
#Green: #00BA38 #Red: #F8766D #Blue: #619CFF
col <-  c("#6495ED", "#CD0000","#006400")
col <- col[as.numeric(tpcagr$main)]

pdf("sample_cluster_no_replicate/scatterplot_VSD_609samples_01.pdf")
# plot
with(tpcagr, scatterplot3d(PC1, PC2, PC3, color = col ,pch=19, main="PCA-plot for normalized counts (vsd)")) 
# add legend
legend("topleft", pch=19, col=c("#6495ED", "#CD0000","#006400"), legend=c("Primary","Metastatic",
"Normal"))
dev.off()

pdf("sample_cluster_no_replicate/scatterplot_counts.VSD_609samples_02.pdf")
with(tpcagr, scatterplot3d(PC1, PC2, PC3, color =  col, pch=19, main="PCA-plot for normalized counts (vsd)", grid=TRUE, box=FALSE, col.grid = "grey", lty.grid=par("lty"))) 
legend("topleft", pch=19, col=c("#6495ED", "#CD0000","#006400"), legend=c("Primary","Metastatic",
"Normal"))
dev.off()


