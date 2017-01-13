# XGB-TFBSContext

XGB-TFBSContext contains code and notebook for "An Ensemble Approach to Elucidating Transcription Factor Binding Specificity and Occupancy".  The repository includes the following files and folders. 

### Prerequisites

The main dependencies are:
- numpy
- scipy
- pybedtools
- pysam
- pybigwig
- pandas
- scikit-learn
- xgboost
- seaborn

All the can be easily installed to a conda environment using:
```
#create a conda environment
conda env create -f environment.yml
#Activate the environment
source activate <env name>
```

## code
This host the Ipython notebooks for the Machine learning modelling and also for the PBM-DNase modelling of TF binding specificity by reranking, reweighing and background correction. 

Included also is the core module for XGB-TFBSContext. 

## Data
Folder contains some of the data for training the XGBoost model. These include the Clustered DNase data and the genome-wide transcription start sites. It also contains the k-mer count files used by PBM-DNase. 

On the local repository, additional files like the human genome, ChIP-seq peaks, PBM intensity data and *k*-mer scores, and the DNA shape files are included. These are not included here due to the enormous space they take. These should be downloaded separately as described below and in the respective Ipython notebooks. 

## Results

The results from feature importance studies and the plots are stored here. 

## Core modules
Some stand alone modules for feature importance studies are included. 
* all_feats.py: Runs full feature importance by elimination studies
* rerank.py : Main module for improving PBM in vivo prediction by re-reranking.
* test_xgb_svm_gbc_sgd.py : Module for investigating the performance of XGB, Gradient boosting, support vector machines and stochastic gradient descend. 

### Step 1: Install the requirements


### Step 2: Get all the required data in place

DNAShape information downloaded from ftp://rohslab.usc.edu/hg19/
    - hg19.HelT.wig.bw
    - hg19.MGW.wig.bw
    - hg19.ProT.wig.bw
    - hg19.Roll.wig.bw

The original Seed and Wobble algorithms from [PBMAnalysisSuite](http://the_brain.bwh.harvard.edu/PBMAnalysisSuite/index.html)

A modified version of the algorithm to take in *k*-mer frequency counts from http://www.bioinf.ict.ru.ac.za/counts_SnW

An executable motif algorithm based on Gibbs sampling from [hierarchicalANOVA](http://thebrain.bwh.harvard.edu/hierarchicalANOVA/)

### 3. Check the respective Ipython notebooks for further details. 


