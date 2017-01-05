import pandas as pd
import numpy as np
from math import  exp
import seaborn as sns
import glob
import os

#import pybedtools
#import pyBigWig
#import pysam
pd.set_option('display.max_colwidth', -1)

import matplotlib.pyplot as plt
import xgboost as xgb

from XGB_TFBSContext import *

shape_path = "/home/ckibet/lustre/Dream_challenge/DNAShape"
human_genome = "/home/ckibet/lustre/Dream_challenge/annotations"
chipseq_path = "/home/ckibet/lustre/XGB-TFBSContext/Data/Downloaded"


dn_hg_dict, kmer_name = get_kmer_dict_rev("%s/Data/dn_hg_max_normalized.txt" % BASE_DIR, "test")
hg_dn_dict, kmer_name = get_kmer_dict_rev("%s/Data/hg_dn_backround_noise_minmax.txt" % BASE_DIR, "test")


def get_feature_df(tf, pos):
    """
    Given a TF and the position of the peak file of interest
    Creat a DataFrame with all the coordinates
    
    This is the main Feature Vector
    """
    peak_files = get_peak_files(tf)

    combined_bed, trim_to = get_combined_bed(peak_files[pos])

    E_score_dict, kmer_name = get_contigmers_dict(get_contigmers(tf)[0],"test")

    ## Calculate all the necessary features
    #E_score_combined = get_kmer_score(combined_bed, sum_kmer_score, E_score_dict)

    feature_frame = pd.DataFrame()
    feature_frame["sum_kmer_score"] = get_kmer_score(combined_bed, sum_kmer_score, E_score_dict)
    feature_frame ["max_kmer_score"] = get_kmer_score(combined_bed, max_score_kmer, E_score_dict)
    test_score = get_kmer_score(combined_bed, max_score_kmer_pos, E_score_dict)
    double_deal = test_score.apply(pd.Series)
    feature_frame ["max_kmer_score_pos"] = double_deal[0]
    hits_df = get_hits_df(double_deal, combined_bed)
    feature_frame["dnase"] = apply_get_max_dnase(hits_df)
    feature_frame["phatsCons"] = apply_get_phatscon(hits_df)
    feature_frame["phyloP100way"] = apply_get_phatscon(hits_df, "phyloP100way")
    
    feature_frame["dn_hg_score"] = get_kmer_score(combined_bed, max_score_kmer, dn_hg_dict)
    feature_frame["hg_dn_score"] = get_kmer_score(combined_bed, max_score_kmer, hg_dn_dict)
#     feature_frame["pwm_score"] = get_kmer_score(combined_bed, energyscore, get_motif_details(tf))
    feature_frame.reset_index(drop=True, inplace=True)
    pos_tss = get_distance_to_tss(hits_df.head(trim_to))
    neg_tss = get_distance_to_tss(hits_df.tail(trim_to))
    pos_neg_tss = pos_tss.append(neg_tss)
    pos_neg_tss.reset_index(drop=True, inplace=True) 
    feature_frame["tss_dist"] = pos_neg_tss
    for shape in "ProT MGW HelT Roll".split():
        #feature_frame["%s_shape" % shape] = apply_get_shape(hits_df, shape)
        feature_fr = apply_get_full_shape(hits_df).apply(pd.Series)
        feature_fr.columns = get_shape_names(shape)
        feature_frame = feature_frame.T.append(feature_fr.T).T
    return feature_frame, trim_to


def pop_this(feat):
    try:
        all_feats.pop(all_feats.index(feat))
    except ValueError:
        try:
            for i in range(8):
                all_feats.pop(all_feats.index(feat+"_%i" % i))
        except ValueError:
            pass



feat_list = ['max_kmer_score','dnase','sum_kmer_score',"phatsCons",
 'Roll', 'ProT', 'MGW', 'HelT',
 'max_kmer_score_pos','dn_hg_score',
 'hg_dn_score',"tss_dist", "phyloP100way"]
        
## test for feature importance by leav one out elimination
tf_in_pbm_chip = ['Ap2',
 'Arid3a',
 'Egr1',
 'Elk1',
 'Elk4',
 'Ets1',
 'Gabp',
 'Gata3',
 'Gr',
 'Hnf4a',
 'Irf3',
 'Jund',
 'Mafk',
 'Max',
 'Pou2f2',
 'Rxra',
 'Sp1',
 'Srf',
 'Tbp',
 'Tcf7l2']



with open("%s/Results/test.txt" % BASE_DIR, "w") as tf_scores:
    
    tf_scores.write("Tf_name\t")
    for j in "sgd, svm, xgbb, gradient".split():
        tf_scores.write("%s\t" % j)
    for tf in tf_in_pbm_chip:
        tf_scores.write("\n%s\t" % tf)
        #tf_feats.write("\n%s\t" % tf)
        print tf
        pybedtools.cleanup()
        
        #feature_frame, trim_to = get_feature_df(tf, 0)
        #feature_frame_p,trim_to_p =  get_feature_df(tf, -1)
        #y_train = np.concatenate((np.ones(trim_to), np.zeros(trim_to)), axis=0)
        #y_test = np.concatenate((np.ones(trim_to_p), np.zeros(trim_to_p)), axis=0)
        
        #all_feats = list(feature_frame.columns)
        
        #All
#         my_model = train_xgboost(feature_frame[all_feats], y_train, tf)
#         testdmat = xgb.DMatrix(feature_frame_p[all_feats], y_test)
#         y_pred = my_model.predict(testdmat)

        
        
        
        feature_frame = feature_frame.fillna(0)
        feature_frame_p = feature_frame_p.fillna(0)
        
        sgd = train_sgd(feature_frame, feature_frame_p, y_train, y_test)
        svms = train_svm(feature_frame, feature_frame_p, y_train, y_test)
        xgb = train_xgb(feature_frame, feature_frame_p, y_train, y_test)
        gradient = train_gradient(feature_frame, feature_frame_p, y_train, y_test)
        
        for mod in [sgd, svms, xgb, gradient]:
            tf_scores.write("%.4f\t" % mod)

