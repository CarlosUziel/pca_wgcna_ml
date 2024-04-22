#####
library("WGCNA")
library(edgeR)
library(DESeq2)
library(matrixStats)
library(data.table)
library(clusterProfiler)
library("org.Hs.eg.db")
# The following setting is important, do not omit.
options(stringsAsFactors = FALSE)

setwd("/Users/rsheibani/Documents/projects/prad/paper_plots_final/paper_v03/wgcna")

###Pipline parameters
###Build the network
#network_file = wgcna_path.joinpath(f"{correlation_type}_{network_type}_network.RDS")

#logging.info("Calculating network...")
#wgcna_args = dict(
#  networkType=network_type,
#  corType=correlation_type,
#  power=power,
#  maxBlockSize=35000,
#  minModuleSize=30,
#  reassignThreshold=1e-6,
#  detectCutHeight=0.998,
#  mergeCutHeight=0.15,
#  deepSplit=2,
#  numericLabels=True,
#  pamStage=True,
#  pamRespectsDendro=True,
#  verbose=0,
#)
##
###############################
#upload data from PRIM_vs_NORM Figure6 A and C
################################

sft <- readRDS("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/bicor_signed_sft.RDS")
net_concept <- readRDS("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/signed_network_concepts.RDS")
net <- readRDS("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/bicor_signed_network.RDS")
datExpr <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/vst_net.csv",sep=",", header=T, check.names = FALSE, stringsAsFactors=FALSE)

str(datExpr)
#'data.frame':	552 obs. of  1407 variables

#Convert GeneName col to Num and remove the colname
datExpr2 <- data.table(datExpr)
datExpr2$V1 <- as.numeric(as.factor(datExpr2$V1))

str(datExpr2)

datExpr3 = as.data.frame(datExpr2[, -c(1)]) 

##########Performing topological overlap
##A soft threshold of 10 was chosen based on the scale independence plot beginning to finally level out at that point for good. The dip from 8 to 9 prevented me from choosing 8. We choose a topology cut-off of 0.80, which is arbitary but similar to the number suggested in WGCNA tutorials (see “A General Framework for Weighted Gene CoExpression Network Analysis”, 2005).

# Choosing a power of 10 based on Scale independence plot and 0.8 threshold
power=sft$powerEstimate #4
power
#[1] 9

softPower <- 9

pickSoftThreshold(datExpr3, powerVector=powers, networkType = "signed", verbose = 2) # calculate soft threshold for signed network

adj = adjacency(datExpr3, power = power, type="signed") # build adjacency matrix signed

distTOM = TOMdist(adj, TOMType = "signed") # calculate dissimilarity TOM signed
###
# Call the hierarchical clustering function
geneTree <- hclust(as.dist(distTOM), method = "average")
##
# We like large modules, so we set the minimum module size relatively high:
minModuleSize <- 30 

#######Relationships between modules
##Now that we have identified modules in our data, we want to explore the relationships between those modules a little more. Note that grey and turquoise modules were removed in these analyses because turquoise contained all genes that did not get slotted in to a specific module and grey had only 4 genes (below our set module size)

# Calculate eigengenes for each module
MEs <- net$MEs
str(net$MEs)
#'data.frame':	552 obs. of  7 variables:
#$ ME6: num  -0.0208 -0.0221 0.0754 -0.0318 -0.0191 ...
#$ ME4: num  -0.0288 -0.0306 0.0302 0.0475 -0.0403 ...
#$ ME3: num  -0.0382 0.0219 0.0259 0.0148 -0.0231 ...
#$ ME1: num  0.01372 0.00545 0.00804 0.05724 0.01052 ...
#$ ME2: num  0.000114 0.006092 -0.018073 0.015038 0.030387 ...
#$ ME5: num  -0.01516 -0.00946 -0.00224 0.04227 0.01704 ...
#$ ME0: num  -0.01151 -0.01328 -0.02097 0.00552 -0.01008 ...

# Calculate dissimilarity of module eigengenes
MEDiss <- 1-cor(MEs)

# Cluster module eigengenes
METree <- hclust(as.dist(MEDiss), method = "average")

##
dynamicColors <- labels2colors(net$colors)
table(dynamicColors)
#dynamicColors
#blue     brown     green      grey       red turquoise    yellow 
#224       204       111       371        38       256       202 

##
pdf(file = "sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0_standard_dendrogram_Dynamic_Tree_ff.pdf")  
plotDendroAndColors(geneTree, dynamicColors, "Dynamic Tree Cut", 
                    dendroLabels = FALSE, hang = 0.03, addGuide = TRUE, 
                    guideHang = 0.04, main = "Gene dendrogram and module colors (prim/norm)")
dev.off()

pdf(file = "sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0_standard_dendrogram_modulecolors_Tree_f.pdf")  
plotDendroAndColors(net$dendrograms[[1]], dynamicColors[net$blockGenes[[1]]],
                    "Module colors",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05, main = "Cluster dendrogram (prim/norm)")
dev.off()

#######
# Call an automatic merging function
merge <- mergeCloseModules(datExpr3, dynamicColors, cutHeight = 0.10, verbose = 3)

str(merge)
#List of 7
#$ colors   : chr [1:1406] "grey" "blue" "blue" "brown" ...
#$ dendro   :List of 7
#..$ merge      : int [1:5, 1:2] -5 -4 -3 -2 -1 -6 1 2 3 4
#..$ height     : num [1:5] 0.186 0.435 0.594 0.78 0.856
#..$ order      : int [1:6] 1 2 3 4 5 6
#..$ labels     : chr [1:6] "MEred" "MEyellow" "MEbrown" "MEturquoise" ...
#..$ method     : chr "average"
#..$ call       : language fastcluster::hclust(d = as.dist(ConsDiss), method = "average")
#..$ dist.method: NULL
#..- attr(*, "class")= chr "hclust"
#$ oldDendro:List of 7
#..$ merge      : int [1:5, 1:2] -5 -4 -3 -2 -1 -6 1 2 3 4
#..$ height     : num [1:5] 0.186 0.435 0.594 0.78 0.856
#..$ order      : int [1:6] 1 2 3 4 5 6
#..$ labels     : chr [1:6] "MEred" "MEyellow" "MEbrown" "MEturquoise" ...
#..$ method     : chr "average"
#..$ call       : language fastcluster::hclust(d = as.dist(ConsDiss), method = "average")
#..$ dist.method: NULL
#..- attr(*, "class")= chr "hclust"
#$ cutHeight: num 0.1
#$ oldMEs   :'data.frame':	552 obs. of  6 variables:
#  ..$ MEred      : num [1:552] -0.0208 -0.0221 0.0754 -0.0318 -0.0191 ...
#..$ MEyellow   : num [1:552] -0.0288 -0.0306 0.0302 0.0475 -0.0403 ...
#..$ MEbrown    : num [1:552] -0.0382 0.0219 0.0259 0.0148 -0.0231 ...
#..$ MEturquoise: num [1:552] 0.01372 0.00545 0.00804 0.05724 0.01052 ...
#..$ MEblue     : num [1:552] 0.000114 0.006092 -0.018073 0.015038 0.030387 ...
#..$ MEgreen    : num [1:552] -0.01516 -0.00946 -0.00224 0.04227 0.01704 ...
#$ newMEs   :'data.frame':	552 obs. of  7 variables:
#  ..$ MEred      : num [1:552] -0.0208 -0.0221 0.0754 -0.0318 -0.0191 ...
#..$ MEyellow   : num [1:552] -0.0288 -0.0306 0.0302 0.0475 -0.0403 ...
#..$ MEbrown    : num [1:552] -0.0382 0.0219 0.0259 0.0148 -0.0231 ...
#..$ MEturquoise: num [1:552] 0.01372 0.00545 0.00804 0.05724 0.01052 ...
#..$ MEblue     : num [1:552] 0.000114 0.006092 -0.018073 0.015038 0.030387 ...
#..$ MEgreen    : num [1:552] -0.01516 -0.00946 -0.00224 0.04227 0.01704 ...
#..$ MEgrey     : num [1:552] -0.01151 -0.01328 -0.02097 0.00552 -0.01008 ...
#$ allOK    : logi TRUE

# Extracting the merged module colors
moduleColors <- merge$colors

# Constructing numerical labels corresponding to the colors
colorOrder = c("grey", standardColors(50))
#;
moduleLabels = match(moduleColors, colorOrder)-1

################Gene enrichment analyses
#########Now that we had explored the modules relationship to each other, we wanted to see if the genes within each module were enriched for gene ontology, KEGG, and Reactome terms. Not all genes map through ‘bitr’ so warnings were supressed

# Making a list of module names
modNames <- substring(names(merge$oldMEs), 3)

# Correlating each genes expression profile with the module eigengenes in order to create module gene sets
geneModuleMembership <- as.data.frame(cor(datExpr3, merge$oldMEs, use = "p"))
# "For each module, we also define a quantitative measure of module membership MM as the correlation of the module eigengene and the gene expression profile." - WGCNA tutorial

# Iteratively creating a list of module genesets to test. These are in ensembl ids

moduleGeneSets<-lapply((modNames),function(module){
  column = match(module, (modNames))
  moduleGenes = moduleColors==module
  rownames(geneModuleMembership[moduleGenes,])
})
names(moduleGeneSets)<-modNames
str(moduleGeneSets)
#List of 6
#$ red      : chr [1:38] "153643" "25876" "201625" "387885" ...
#$ yellow   : chr [1:202] "4646" "4583" "84059" "283932" ...
#$ brown    : chr [1:204] "653820" "11004" "9768" "284076" ...
#$ turquoise: chr [1:256] "85028" "401466" "222967" "27040" ...
#$ blue     : chr [1:224] "153571" "100506826" "6300" "345930" ...
#$ green    : chr [1:111] "79874" "79751" "64118" "100506082" ...

saveRDS(moduleGeneSets, "sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/moduleGeneSets_oldMEs.RData")

# Looking at reactome enrichment
cr<-compareCluster(geneCluster=moduleGeneSets,fun="enrichPathway", organism="human", pvalueCutoff = 1, qvalueCutoff  = 1, readable = T)

saveRDS(cr, "sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/compareCluster_oldMEs.RData")

str(cr)
#..@ geneClusters        :List of 6
#.. ..$ red      : chr [1:38] "153643" "25876" "201625" "387885" ...
#.. ..$ yellow   : chr [1:202] "4646" "4583" "84059" "283932" ...
#.. ..$ brown    : chr [1:204] "653820" "11004" "9768" "284076" ...
#.. ..$ turquoise: chr [1:256] "85028" "401466" "222967" "27040" ...
#.. ..$ blue     : chr [1:224] "153571" "100506826" "6300" "345930" ...
#.. ..$ green    : chr [1:111] "79874" "79751" "64118" "100506082" ...

cr <- setReadable(cr, OrgDb = "org.Hs.eg.db", keyType="ENTREZID")
head(cr) 

write.table(as.data.frame(cr), file="sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0_standard_enrichPathway_oldMEs.tsv", quote=F, sep="\t", row.names=T, col.names=T)

pdf("sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__PRIM_vs_NORM_padj_0_05_up_1_0_standard_enrichPathway_oldMEs_5.pdf", width = 15, height=10)
dotplot(cr ,showCategory=5, label_format = 100)+ggtitle("Reactome geneset enrichment by module (prim/norm)")
dev.off()

################################
###upload data from MET_BB_vs_PRIM Figure 7 D and F
################################
sft <- readRDS("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/bicor_signed_sft.RDS")
net_concept <- readRDS("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/signed_network_concepts.RDS")
net <- readRDS("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/bicor_signed_network.RDS")
datExpr <- read.table("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/vst_net.csv",sep=",", header=T, check.names = FALSE, stringsAsFactors=FALSE)

str(datExpr)
#'data.frame':	557 obs. of  3806 variables:

#Convert GeneName col to Num and remove the colname
datExpr2 <- data.table(datExpr)
datExpr2$V1 <- as.numeric(as.factor(datExpr2$V1))

str(datExpr2)

datExpr3 = as.data.frame(datExpr2[, -c(1)]) 

##########Performing topological overlap
##A soft threshold of 10 was chosen based on the scale independence plot beginning to finally level out at that point for good. The dip from 8 to 9 prevented me from choosing 8. We choose a topology cut-off of 0.80, which is arbitary but similar to the number suggested in WGCNA tutorials (see “A General Framework for Weighted Gene CoExpression Network Analysis”, 2005).

# Choosing a power of 10 based on Scale independence plot and 0.8 threshold
power=sft$powerEstimate #4
power
#[1] 10

softPower <- 10

pickSoftThreshold(datExpr3, powerVector=powers, networkType = "signed", verbose = 2) # calculate soft threshold for signed network

adj = adjacency(datExpr3, power = power, type="signed") # build adjacency matrix signed

distTOM = TOMdist(adj, TOMType = "signed") # calculate dissimilarity TOM signed

# Call the hierarchical clustering function
geneTree <- hclust(as.dist(distTOM), method = "average")

# We like large modules, so we set the minimum module size relatively high:
minModuleSize <- 30 

#dynamicColors
dynamicColors <- labels2colors(net$colors)
table(dynamicColors)
#dynamicColors
#black      blue     brown     green      grey   magenta      pink       red turquoise    yellow 
#222       389       362       333      1196        59       179       226       503       336 

#######Relationships between modules
##Now that we have identified modules in our data, we want to explore the relationships between those modules a little more. Note that grey and turquoise modules were removed in these analyses because turquoise contained all genes that did not get slotted in to a specific module and grey had only 4 genes (below our set module size)

# Calculate eigengenes for each module
MEs <- net$MEs
str(net$MEs)
#'data.frame':	557 obs. of  10 variables:
#  $ ME3: num  -0.01404 -0.02532 0.00428 -0.01234 0.01093 ...
#$ ME1: num  0.00046 -0.00425 0.01464 -0.01861 0.01527 ...
#$ ME7: num  -0.01372 0.03682 -0.00198 -0.00824 0.01636 ...
#$ ME4: num  -0.03461 0.00206 0.00175 -0.00627 -0.03184 ...
#$ ME6: num  0.02988 -0.00355 0.01396 0.01013 -0.00123 ...
#$ ME8: num  0.000238 -0.00494 0.008875 -0.011475 -0.023803 ...
#$ ME9: num  -0.0435 -0.0122 -0.0132 0.0267 -0.0141 ...
#$ ME2: num  0.028659 -0.007793 0.000637 0.027334 -0.004273 ...
#$ ME5: num  0.00843 -0.0168 -0.01594 0.02382 -0.00506 ...
#$ ME0: num  -0.021 -0.0266 -0.0286 -0.0117 -0.0289 ..

# Calculate dissimilarity of module eigengenes
MEDiss <- 1-cor(MEs)

# Cluster module eigengenes
METree <- hclust(as.dist(MEDiss), method = "average")
##
pdf(file = "sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0_standard_dendrogram_Dynamic_Tree_f.pdf")  
plotDendroAndColors(geneTree, dynamicColors, "Dynamic Tree Cut", 
                    dendroLabels = FALSE, hang = 0.03, addGuide = TRUE, 
                    guideHang = 0.04, main = "Gene dendrogram and module colors (met/prim)")
dev.off()

pdf(file = "sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0_standard_dendrogram_modulecolors_Tree_f.pdf")  
plotDendroAndColors(net$dendrograms[[1]], dynamicColors[net$blockGenes[[1]]],
                    "Module colors",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05, main = "Cluster dendrogram (met/prim)")
dev.off()

######## Call an automatic merging function
merge <- mergeCloseModules(datExpr3, dynamicColors, cutHeight = 0.10, verbose = 3)
str(merge)
#List of 7
#$ colors   : chr [1:3805] "grey" "green" "black" "green" ...
#$ dendro   :List of 7
#..$ merge      : int [1:8, 1:2] -2 -8 -5 -1 -4 -7 5 4 -3 -9 ...
#..$ height     : num [1:8] 0.141 0.146 0.222 0.299 0.403 ...
#..$ order      : int [1:9] 1 2 3 4 5 6 7 8 9
#..$ labels     : chr [1:9] "MEbrown" "MEblack" "MEturquoise" "MEyellow" ...
#..$ method     : chr "average"
#..$ call       : language fastcluster::hclust(d = as.dist(ConsDiss), method = "average")
#..$ dist.method: NULL
#..- attr(*, "class")= chr "hclust"
#$ oldDendro:List of 7
#..$ merge      : int [1:8, 1:2] -2 -8 -5 -1 -4 -7 5 4 -3 -9 ...
#..$ height     : num [1:8] 0.141 0.146 0.222 0.299 0.403 ...
#..$ order      : int [1:9] 1 2 3 4 5 6 7 8 9
#..$ labels     : chr [1:9] "MEbrown" "MEblack" "MEturquoise" "MEyellow" ...
#..$ method     : chr "average"
#..$ call       : language fastcluster::hclust(d = as.dist(ConsDiss), method = "average")
#..$ dist.method: NULL
#..- attr(*, "class")= chr "hclust"
#$ cutHeight: num 0.1
#$ oldMEs   :'data.frame':	557 obs. of  9 variables:
#  ..$ MEbrown    : num [1:557] -0.01404 -0.02532 0.00428 -0.01234 0.01093 ...
#..$ MEblack    : num [1:557] -0.01372 0.03682 -0.00198 -0.00824 0.01636 ...
#..$ MEturquoise: num [1:557] 0.00046 -0.00425 0.01464 -0.01861 0.01527 ...
#..$ MEyellow   : num [1:557] -0.03461 0.00206 0.00175 -0.00627 -0.03184 ...
#..$ MEpink     : num [1:557] 0.000238 -0.00494 0.008875 -0.011475 -0.023803 ...
#..$ MEred      : num [1:557] 0.02988 -0.00355 0.01396 0.01013 -0.00123 ...
#..$ MEmagenta  : num [1:557] -0.0435 -0.0122 -0.0132 0.0267 -0.0141 ...
#..$ MEblue     : num [1:557] 0.028659 -0.007793 0.000637 0.027334 -0.004273 ...
#..$ MEgreen    : num [1:557] 0.00843 -0.0168 -0.01594 0.02382 -0.00506 ...
#$ newMEs   :'data.frame':	557 obs. of  10 variables:
#  ..$ MEbrown    : num [1:557] -0.01404 -0.02532 0.00428 -0.01234 0.01093 ...
#..$ MEblack    : num [1:557] -0.01372 0.03682 -0.00198 -0.00824 0.01636 ...
#..$ MEturquoise: num [1:557] 0.00046 -0.00425 0.01464 -0.01861 0.01527 ...
#..$ MEyellow   : num [1:557] -0.03461 0.00206 0.00175 -0.00627 -0.03184 ...
#..$ MEpink     : num [1:557] 0.000238 -0.00494 0.008875 -0.011475 -0.023803 ...
#..$ MEred      : num [1:557] 0.02988 -0.00355 0.01396 0.01013 -0.00123 ...
#..$ MEmagenta  : num [1:557] -0.0435 -0.0122 -0.0132 0.0267 -0.0141 ...
#..$ MEblue     : num [1:557] 0.028659 -0.007793 0.000637 0.027334 -0.004273 ...
#..$ MEgreen    : num [1:557] 0.00843 -0.0168 -0.01594 0.02382 -0.00506 ...
#..$ MEgrey     : num [1:557] -0.021 -0.0266 -0.0286 -0.0117 -0.0289 ...
#$ allOK    : logi TRUE


######### Extracting the merged module colors
moduleColors <- merge$colors

# Constructing numerical labels corresponding to the colors
colorOrder = c("grey", standardColors(50))

moduleLabels = match(moduleColors, colorOrder)-1

################Gene enrichment analyses
#########Now that we had explored the modules relationship to each other, we wanted to see if the genes within each module were enriched for gene ontology, KEGG, and Reactome terms. Not all genes map through ‘bitr’ so warnings were supressed

# Making a list of module names
modNames <- substring(names(merge$oldMEs), 3)

# Correlating each genes expression profile with the module eigengenes in order to create module gene sets
geneModuleMembership <- as.data.frame(cor(datExpr3, merge$oldMEs, use = "p"))
# "For each module, we also define a quantitative measure of module membership MM as the correlation of the module eigengene and the gene expression profile." - WGCNA tutorial

# Iteratively creating a list of module genesets to test. These are in ensembl ids
moduleGeneSets<-lapply((modNames),function(module){
  column = match(module, (modNames))
  moduleGenes = moduleColors==module
  rownames(geneModuleMembership[moduleGenes,])
})
names(moduleGeneSets)<-modNames
str(moduleGeneSets)
#List of 9
#$ brown    : chr [1:362] "115352" "8514" "6691" "2357" ...
#$ black    : chr [1:222] "102723641" "728978" "9002" "105373244" ...
#$ turquoise: chr [1:503] "2191" "26045" "115265" "10203" ...
#$ yellow   : chr [1:336] "91057" "10940" "8329" "79740" ...
#$ pink     : chr [1:179] "55356" "342918" "199953" "7592" ...
#$ red      : chr [1:226] "7571" "101927761" "100652865" "389333" ...
#$ magenta  : chr [1:59] "6232" "10603" "3177" "3995" ...
#$ blue     : chr [1:389] "79829" "284402" "441476" "440894" ...
#$ green    : chr [1:333] "162968" "84787" "730227" "146705" ...

saveRDS(moduleGeneSets, "sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/moduleGeneSets_oldMEs.RData")

cr<-compareCluster(geneCluster=moduleGeneSets,fun="enrichPathway", organism="human", pvalueCutoff = 1, qvalueCutoff  = 1, readable = T)

saveRDS(cr, "sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/compareCluster_oldMEs.RData")

str(cr)
#..@ geneClusters        :List of 9
#.. ..$ brown    : chr [1:362] "115352" "8514" "6691" "2357" ...
#.. ..$ black    : chr [1:222] "102723641" "728978" "9002" "105373244" ...
#.. ..$ turquoise: chr [1:503] "2191" "26045" "115265" "10203" ...
#.. ..$ yellow   : chr [1:336] "91057" "10940" "8329" "79740" ...
#.. ..$ pink     : chr [1:179] "55356" "342918" "199953" "7592" ...
#.. ..$ red      : chr [1:226] "7571" "101927761" "100652865" "389333" ...
#.. ..$ magenta  : chr [1:59] "6232" "10603" "3177" "3995" ...
#.. ..$ blue     : chr [1:389] "79829" "284402" "441476" "440894" ...
#.. ..$ green    : chr [1:333] "162968" "84787" "730227" "146705" ...

cr <- setReadable(cr, OrgDb = "org.Hs.eg.db", keyType="ENTREZID")
head(cr) 

write.table(as.data.frame(cr), file="sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0_standard_enrichPathway_oldMEs.tsv", quote=F, sep="\t", row.names=T, col.names=T)

pdf("sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0/standard/sample_cluster_no_replicates_MET_BB+NORM+PRIM__MET_BB_vs_PRIM_padj_0_05_up_1_0_standard_enrichPathway_oldMEs_5.pdf", width = 15, height=10)
dotplot(cr ,showCategory=5, label_format = 100)+ggtitle("Reactome geneset enrichment by module (met/prim)")
dev.off()





