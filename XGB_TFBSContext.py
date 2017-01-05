from multiprocessing import Pool, cpu_count
import subprocess
import pandas as pd
import numpy as np
from math import  exp
import seaborn as sns
import glob
import os

import pybedtools
import pyBigWig
import pysam
pd.set_option('display.max_colwidth', -1)

BASE_DIR = "/home/ckibet/lustre/XGB-TFBSContext"

shape_path = "/home/ckibet/lustre/Dream_challenge/DNAShape"
human_genome = "/home/ckibet/lustre/Dream_challenge/annotations"
chipseq_path = "/home/ckibet/lustre/XGB-TFBSContext/Data/Downloaded"


revcompl = lambda x: ''.join([{'A':'T','C':'G','G':'C','T':'A'}[B] for B in x][::-1])

def score_kmer(kmer,kmerdict,revcompl):
    score=0
    if kmer in kmerdict:
        score=float(kmerdict[kmer])
    else:
        kmer2=revcompl(kmer)
        score=float(kmerdict[kmer2])
    return score

def sum_kmer_score(kmerdict, seq):
    k_mers = find_kmers(seq, 8)
    tot_score = 0
    for kmer in k_mers:
        if kmer in kmerdict:
            score = float(kmerdict[kmer])
        else:
            score = 0.0
            #kmer2 = revcompl(kmer)
            #score = float(kmerdict[kmer2])
        tot_score += score
    return tot_score

def max_score_kmer(kmerdict, seq):
    """
    Score the k-mers by the maximum score
    """
    
    k_mers = find_kmers(seq, 8)
    tot_score = []
    for kmer in k_mers:
        if kmer in kmerdict:
            score = float(kmerdict[kmer])
        else:
            score = 0.0
            #kmer2 = revcompl(kmer)
            #score = float(kmerdict[kmer2])
        tot_score.append(score)
    max_pos = tot_score.index(max(tot_score))
    
    return max(tot_score)


def max_score_kmer_pos(kmerdict, seq):
    k_mers = find_kmers(seq, 8)
    tot_score = []
    for kmer in k_mers:
        if kmer in kmerdict:
            score = float(kmerdict[kmer])
        else:
            score = 0.0
            #kmer2 = revcompl(kmer)
            #score = float(kmerdict[kmer2])
        tot_score.append(score)
    max_pos = tot_score.index(max(tot_score))
    
    return sum(tot_score[max_pos-4:max_pos+4]), max_pos-4


def energyscore(pwm_dictionary, seq):
    """
    Score sequences using the beeml energy scoring approach.

    Borrowed greatly from the work of Zhao and Stormo

    P(Si)=1/(1+e^Ei-u)

    Ei=sumsum(Si(b,k)e(b,k))

    Previous approaches seem to be using the the minimum sum of the
    energy contribution of each of the bases of a specific region.

    This is currently showing some promise but further testing is
    needed to ensure that I have a robust algorithm.
    """
    
    energy_list = []
    pwm_length = len(pwm_dictionary["A"])
    pwm_dictionary_rc = rc_pwm(pwm_dictionary, pwm_length)
    for i in range(len(seq) - 1):
        energy = 0
        energy_rc = 0
        for j in range(pwm_length - 1):
            if (j + i) >= len(seq):
                energy += 0.25
                energy_rc += 0.25
            else:
                energy += pwm_dictionary[seq[j + i]][j]
                energy_rc += pwm_dictionary_rc[seq[j + i]][j]

            energy_list.append(1 / (1 + (exp(energy))))
            energy_list.append(1 / (1 + (exp(energy_rc))))
    energy_score = min(energy_list)
    return energy_score

# def energy_score_kmer(seq,kmerdict,revcompl):
#     k_mers=find_kmers(seq,8)
#     tot_score = 0
#     for kmer in k_mers:
#         if kmer in kmerdict:
#             score=float(kmerdict[kmer])
#         else:
#             kmer2=revcompl(kmer)
#             score=float(kmerdict[kmer2])
#         tot_score+=score
#     return tot_score

def find_kmers(string, kmer_size):
    kmers = []
    for i in range(0, len(string)-kmer_size+1):
        kmers.append(string[i:i+kmer_size])
    return kmers

def getKey(item):
    return item[1]

def get_kmer_dict(kmerscore, kmer_name):
    scoredict={}
    with open(kmerscore) as kmers:
        for line in kmers:
            ke,rem, val=line.split()
            scoredict[ke]=val
    return scoredict, kmer_name

def get_kmer_dict_rev(kmerscore, kmer_name):
    """
    Convert the forward and reverse kmers into a dictionary
    for a quick look-up and scoring of sequences
    """
    test = pd.read_table(kmerscore, index_col="8-mer", usecols=["8-mer", "E-score"])
    test.fillna(0, inplace=True)
    test2 = pd.read_table(kmerscore, index_col="8-mer.1", usecols=["8-mer.1", "E-score"])
    test2.index.name = "8-mer"
    test2.fillna(0, inplace=True)
    combined = test.append(test2)
    combined_dict = combined.to_dict()["E-score"]

    return combined_dict, kmer_name


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# Sequence scoring  in parallel  
def score_from_genome(bed_df):
    
    return bed_df.apply(lambda row: fetch_and_score_seq(row[0], row[1], row[2]), axis=1)

def fetch_and_score_seq(contig, start, end):
    genome = pysam.FastaFile('%s/hg19.genome.fa' % human_genome)
    return score_function(pwm_dictionary, genome.fetch(contig, start, end).upper())


# def score_from_genome_shape(bed_df):
    
#     return bed_df.apply(lambda row: fetch_and_score_seq_shape(row[0], row[1], row[2]), axis=1)

# def fetch_and_score_seq_shape(contig, start, end):
#     genome = pysam.FastaFile('/home/kipkurui/Dream_challenge/annotations/hg19.genome.fa')
#     return score_function(pwm_dictionary, genome.fetch(contig, start, end).upper())

def get_full_shape(shape_file, ch, start, end):
    """
    Extract the shape information from the full length TFBS
    as opposed to mean
    """
    bw = pyBigWig.open(shape_file)
    
    return bw.values(ch, start, end)

def apply_get_full_shape(bed_df, shape="Roll"):
    """
    Keep it this way for now, but when I have to parallelize,
    I will have to think differently
    """
    shape_file = "%s/hg19.%s.wig.bw" % (shape_path, shape)
    test = bed_df.apply(lambda row: get_full_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape

def get_mean_shape(shape_file, ch, start, end):
    """
    Extract the maximum fold enrichment from Bigwig files
    
    Keep in mind this error:
    "An error occurred while fetching values!"
    
    This error lead to incorrect results. 
    Will have to re-think way out latter
    
    SOLVED: The file cannot be accessed concerrently. 
    Should open a new file handle
    for each run. 
    """
    bw = pyBigWig.open(shape_file)
    try:
        return np.mean(bw.values(ch, start, end))
    except RuntimeError:
        return 0

    
def apply_get_shape(bed_df, shape="Roll"):
    """
    Keep it this way for now, but when I have to parallelize,
    I will have to think differently
    """
    shape_file = "%s/DNAShape/hg19.%s.wig.bw" % (shape_path, shape)
    test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape

##Leave these here for a larger scale analsysis

# def get_mean_shape(ch, start, end):
#     """
#     Extract the maximum fold enrichment from Bigwig files
    
#     Keep in mind this error:
#     "An error occurred while fetching values!"
    
#     This error lead to incorrect results. 
#     Will have to re-think way out latter
    
#     SOLVED: The file cannot be accessed concerrently. 
#     Should open a new file handle
#     for each run. 
#     """
#     bw = pyBigWig.open("/home/kipkurui/Dream_challenge/DNAShape/hg19.%s.wig.bw" % shape)
#     try:
#         return np.mean(bw.values(ch, start, end))
#     except RuntimeError:
#         return 0

# def apply_get_shape(bed_df):
#     """
#     Get max DNase fold enrichment over the whole
#     dataframe using pandas apply function
#     """
#     test = bed_df.apply(lambda row: get_mean_shape(row[0], row[1], row[2]), axis=1)
    
#     mean_shape = test.fillna(0)
    
#     return mean_shape


# def apply_get_shape_Roll(bed_df):
#     """
    
#     """
#     shape_file = "%s/hg19.%s.wig.bw" % (shape_path,shape)
#     test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
#     mean_shape = test.fillna(0)
    
#     return mean_shape

# def apply_get_shape_HelT(bed_df):
#     """
    
#     """
#     shape_file = "%s/DNAShape/hg19.HelT.wig.bw" % BASE_DIR
#     test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
#     mean_shape = test.fillna(0)
    
#     return mean_shape

# def apply_get_shape_ProT(bed_df):
#     """
    
#     """
#     shape_file = "%s/DNAShape/hg19.ProT.wig.bw" % BASE_DIR
#     test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
#     mean_shape = test.fillna(0)
    
#     return mean_shape

# def apply_get_shape_MGW(bed_df):
#     """
    
#     """
#     shape_file = "%s/DNAShape/hg19.MGW.wig.bw" % BASE_DIR
#     test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
#     mean_shape = test.fillna(0)
    
#     return mean_shape

def get_max_dnase(ch, start, end):
    """
    Extract the maximum fold enrichment from Bigwig files
    
    Keep in mind this error:
    "An error occurred while fetching values!"
    
    This error lead to incorrect results. 
    Will have to re-think way out latter
    
    SOLVED: The file cannot be accessed concerrently. 
    Should open a new file handle
    for each run. 
    """
    bw = pyBigWig.open("%s/Data/DNase/wgEncodeRegDnaseClusteredV3.bigwig" % BASE_DIR)
    try:
        return np.mean(bw.values(ch, start, end))
    except RuntimeError:
        return 0

def apply_get_max_dnase(bed_df):
    """
    Get max DNase fold enrichment over the whole
    dataframe using pandas apply function
    """
    test = bed_df.apply(lambda row: get_max_dnase(row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape

def get_max_phatscon(con_file, ch, start, end):
    """
    Extract the maximum fold enrichment from Bigwig files
    
    Keep in mind this error:
    "An error occurred while fetching values!"
    
    This error lead to incorrect results. 
    Will have to re-think way out latter
    
    SOLVED: The file cannot be accessed concerrently. 
    Should open a new file handle
    for each run. 
    """
    bw = pyBigWig.open(con_file)
    try:
        return np.mean(bw.values(ch, start, end))
    except RuntimeError:
        return 0

def apply_get_phatscon(bed_df, con_type="phastCons"):
    """
    Get max DNase fold enrichment over the whole
    dataframe using pandas apply function
    """
    con_file = "%s/Data/Conservation/hg19.100way.%s.bw" % (BASE_DIR,con_type)
    test = bed_df.apply(lambda row: get_max_phatscon(con_file, row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape

def get_contigmers_dict(congtigmer, kmer_name):
    """
    Convert the forward and reverse kmers into a dictionary
    for a quick look-up and scoring of sequences
    """
    test = pd.read_table(congtigmer, header=None,index_col=0, usecols=[0,1,3])
    test.columns = [["8-mers", "E-score"]]
    test.index.name = "8-mers"
    combined = test.set_index("8-mers").append(test.drop("8-mers", 1))
    combined_dict = combined.to_dict()["E-score"]

    return combined_dict, kmer_name

def insensitive_glob(pattern):
    """
    Borrowed from answer here: 
    http://stackoverflow.com/questions/8151300/ignore-case-in-glob-on-linux
    """
    def either(c):
        return '[%s%s]'%(c.lower(),c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either,pattern)))

def get_contigmers(tf):
    
    contig_path = "%s/Data/PBM_2016/All_Contig8mers/*/*%s*" % (BASE_DIR,tf.capitalize())
    return insensitive_glob(contig_path)

def get_peak_files(tf):
    peak_path = "%s/*%s*" % (chipseq_path,tf.capitalize())
    
    return glob.glob(peak_path)
    #return insensitive_glob(peak_path)

def add_best8(max_list):
    max_pos = max_list.index(max(max_list))
    sum(maxi[max_pos-4:max_pos+4])
    
def get_bed_from_peaks(peak, width, downstream_distance):
    """
    Given a ChIp-seq peak file, process the peaks to
    a given length and extract the possitive and 
    negative set of a given size
    
    """
    #Read the narrow peak file into a pandas DataFrame
    
    peak_file = pd.read_table(peak, header=None)[[0,1,2]]
    
    #Lets widden the coordinates to 100bp centered around the center
    mid = (peak_file[2] + peak_file[1])/2
    peak_file[1] = (mid - width/2+0.5).apply(int)
    peak_file[2]  = (mid + width/2+0.5).apply(int)
    
    #Extract the negative set located 500bp downstream
    neg_bed = peak_file.copy(deep=True)
    
    neg_bed[1] = neg_bed[1]+downstream_distance
    neg_bed[2] = neg_bed[2]+downstream_distance
    
    # Eliminate repeat masked regions from the bed file
    peak_file = remove_repeats(peak_file) #.to_csv(pos_bed_out, index=None, header=None, sep="\t")
    neg_bed = remove_repeats(neg_bed) #.to_csv(neg_bed_out, index=None, header=None, sep="\t")
    
    #hg = "/home/kipkurui/Project/MAT_server/Data/hg19.fa"
    return peak_file, neg_bed

    # uncomment this, if you need Fasta sequences
    #pybedtools.BedTool.from_dataframe(peak_file).sequence(fi=hg,).save_seqs(negfa_out)

def remove_repeats(dfs):
    """
    Takes a bed file dataframe and eliminated bed
    coordinates that fall within the repeat masked sections
    """
    repeats = pd.read_table("%s/Data/repeat_sites.bed" % BASE_DIR, header=None)
    repeats = pybedtools.BedTool.from_dataframe(repeats)
    
    a = pybedtools.BedTool.from_dataframe(dfs)
    
    test = a.subtract(repeats, A=True)
    
    return test.to_dataframe()


def get_combined_bed(peak):
    """
    Extract and combine the positive and negative
    sequences into a single DataFrame
    """
    peak_file, neg_bed = get_bed_from_peaks(peak, 100, 500)

    trim_to = min(len(peak_file), len(neg_bed))
    trim_to = trim_to/2
    pos_bed = peak_file.head(trim_to)
    pos_bed = pos_bed.sort_values(by=["chrom", "start","end"])
    neg_bed = neg_bed.head(trim_to)
    neg_bed = neg_bed.sort_values(by=["chrom", "start","end"])

    combined_bed = pos_bed.append(neg_bed, ignore_index=True)
    
    return combined_bed, trim_to


## Score sequence of interest

def get_kmer_score(combined_bed, score_fun, score_dict):
    """
    Given a bed file, score the sequences using 
    """
    global pwm_dictionary
    global score_function
    
    pwm_dictionary =score_dict
    score_function=score_fun
    
    return score_from_genome(combined_bed)


def get_distance_to_tss(bed):
    """
    Given a bed file, calculate the distance from the 
    midpoint to the nearest TSS
    
    """
    tss = pd.read_table("%s/Data/Tss_hg19_refseq" % BASE_DIR, header=None)
    tss[2] = tss[1] + 1
    tss[1] = tss[1] - 1
    tss = tss.sort_values(by=[0,1,2], kind="mergesort")
    
    bed = bed.sort_values(by=["chrom", "start","end"], kind="mergesort")
    
    tss_obj = pybedtools.BedTool.from_dataframe(tss)
    
    bed_obj = pybedtools.BedTool.from_dataframe(bed)
    
    bed_closest = bed_obj.closest(tss_obj)
    bed_closest = bed_obj.closest(tss_obj, d=True,k=1,t='first')
    bed_closest_df = bed_closest.to_dataframe()
    
    return bed_closest_df["thickStart"]
    
def get_hits_df(double_deal, combined_bed):
    """
    Given BED DataFrame and coordinate of the hit site
    Create a DF with shape hit coordinates
    
    """
    shape_hits = combined_bed.copy()
    shape_hits['start'] = combined_bed['start'] + double_deal[1].apply(int)
    shape_hits['end'] = shape_hits['start'] + 8
    
    return shape_hits

def get_shape_names(shape):
    """
    Label each of the shape features
    """
    name_list = []
    for i in range(8):
        name_list.append("%s_%i" % (shape,i))
        
    return name_list


###
### XGBoost model
###

import pickle
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import precision_recall_curve, roc_auc_score
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as ro

import matplotlib.pyplot as plt

#Accuracy metrics
from sklearn.metrics import accuracy_score, classification_report, auc

# Creating an learning pipeline
from sklearn.pipeline import Pipeline

from sklearn import feature_selection

from sklearn.externals import joblib

#from xgboost import XGBClassifier

import xgboost as xgb

def train_xgboost(dataframe, y, tf):
    """
    Given a feature DF, train a model using the optimized parameters
    
    The parameters are chosen using cross validation
    """
    xgdmat = xgb.DMatrix(dataframe, y) 

    our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':8, 'min_child_weight':1} 

    final_gb = xgb.train(our_params, xgdmat, num_boost_round = 3000)
    
    #save the model for future reference
    #pickle.dump(final_gb, "%s/annotations/%s/%s_xgboost_pick.dat" % (BASE_DIR, tfs, tfs))
    #joblib.dump(final_gb, "%s/annotations/%s/%s_xgboost.dat" % (BASE_DIR, tfs, tfs))
    
    #Creat a feature importance plot
    #plot_feature_importance(final_gb, "%s/Results/%s_features.png" % (BASE_DIR,tf))
    
    return final_gb


def plot_feature_importance(xgb_model, fig_out):
    sns.set(font_scale = 1.5)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    xgb.plot_importance(xgb_model, ax=ax)
    #ax.plot([0,1,2], [10,20,3])
    fig.savefig(fig_out, bbox_inches='tight')   # save the figure to file
    plt.close(fig)

    
def get_auc(y_test, y_pred, lab=1):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=lab)
    return metrics.auc(fpr, tpr)

def scikitlearn_calc_auPRC(y_true, y_score):
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def calc_auPRC(y_true, y_score):
    """Calculate auPRC using the R package 
    
    From DREAM challenge organizers

    """
    ro.globalenv['pred'] = y_score
    ro.globalenv['labels'] = y_true
    return ro.r('library(PRROC); pr.curve(scores.class0=pred, weights.class0=labels)$auc.davis.goadrich')[0]

def recall_at_fdr(y_true, y_score, fdr_cutoff=0.05):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fdr = 1- precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]
