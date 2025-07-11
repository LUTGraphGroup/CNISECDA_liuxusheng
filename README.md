# CNISECDA

CNISECDA∶ Predicting circRNA-disease associations based on ChebNet and Improved Deep Sparse Autoencoder

## Environmental requirements

- python=3.6
- tensorflow==2.6.2
- numpy==1.15.4
- scipy==1.5.2
- xlrd==1.2.0
- torch==1.10.0+cu102
- pytorch-cuda==11.6
- dgl-cuda10.2

## Dataset

The dataset comes from CircR2Disease and  CircR2Disease v2.0.

## Input file

- `circRNA_disease_association_matrix.csv`：The correlation matrix between circRNAS and diseases
- `disease_semantic_similarity.csv`: Disease semantic similarity matrix
- `disease_GIP_similarity.csv`：Disease Gaussian Interaction Profile Kernel (GIP) similarity matrix
- `disease_similarity_fusion_matrix.csv`：A comprehensive similarity matrix that integrates disease semantic similarity and GIP similarity
- `circRNA_functional_similarity.csv`： circRNA functional similarity matrix
- `circRNA_GIP_similarity.csv`： circRNA Gaussian Interaction Profile Kernel (GIP) similarity matrix
- `circRNA_similarity_fusion_matrix.csv`：A comprehensive similarity matrix that integrates circRNA functional similarity and GIP similarity

## Running files

- `Prepare_data.py`:  To calculate the circRNA_similarity_fusion_matrix and disease_similarity_fusion_matrix.csv
- `chebnet_CNN.py:`  Building heterogeneous graphs, multi-order neighborhood structural features, and high-level semantic representations.
- `GCDSAEMDA_CircR2Disease.py/GCDSAEMDA_CircR2Disease_2.0.py`:  Negative CDA selection, improved deep sparse autoencoder dimensionality reduction, CatBooost.
- `IDSAE.py`:  improved deep sparse autoencoder model 、negative CDA selection method  and ROC(PR) curve drawing method 

