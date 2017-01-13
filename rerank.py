#coding: utf-8

# # Improving *in vitro* using *in vivo* data
# 
# The focus of this notebook is to start a fresh outline the whole journey as clearly as can be. Only the relevant sections to the story in my thesis will make it to this notebook from the other one. 
# 
# The following sections should be converted:
# * A quick versioning and source of the algorithms used
# * Source of data
# * The Python requirements
# * The support, separate modules
# * The iterations as folows:
# 
# 
# Some of the main requirements to do this are:
# * The modified Perl code used to incorporate in vivo information
# * The helper scripts used for counting and data conversion
# * The kmer counts previously generated:
#  - Counts in human genome
#  - Count in Dnase data
#  - Frequency difference of that count
# * Then we can directly generate the motifs and test how they are performing
# 
# 
# With the machine elarning angle to the project established, the next stage is to get this working. Improve on it by using a better scoring function, and also do a bit of comparisons. 
# 
# Some action points related top this chapter:
# * Transform debruijn with dn-hg
# * A background noise correction
#     * Re-rank with k-mer frequency count difference (hg-dn, considered as noise)
#     * Jiangs approach

# ### Requirements:
# 1. The original Seed and Wobble algorithms from http://the_brain.bwh.harvard.edu/PBMAnalysisSuite/index.html
# 2. A modified version of the algorithm to take in k-mer frequency counts from http://www.bioinf.ict.ru.ac.za/counts_SnW
# 3. An executable motif algorithm baded on Gibbs sampling from "http://thebrain.bwh.harvard.edu/hierarchicalANOVA/"

# 

# In[1]:

import os
import glob
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import seaborn as sns

pd.set_option('display.max_colwidth', -1)
#get_ipython().magic(u'matplotlib inline')


# ### Import MARSTools for evaluations

# In[23]:

BASE_DIR = "/home/ckibet/lustre/PBM_DNase"
scripts_path = "%s/Scripts" % BASE_DIR

# In[5]:
os.chdir("/home/ckibet/lustre")
from MARSTools import Assess_by_score as score
os.chdir(scripts_path)


# ## Set the base directory and figure path

# In[14]:

BASE_DIR = "/home/ckibet/lustre/PBM_DNase"


# In[13]:

figure_path = "/home/ckibet/lustre/XGB-TFBSContext/Results/Chapter6/Figs"


# ### Get a list of TFs affected by Sticky k-mers
# 
# The Sticky k-mers Identified by Jiang et al and downloadable from: are used . 


pbm_chip = []
pbmchip2name = {}
with open('/home/ckibet/lustre/XGB-TFBSContext/Data/Pbm_Chip_details.txt') as pbmnchip:
    for line in pbmnchip:
        if line.startswith('Tf_id'):
            continue
        else:
            pbm_chip.append(line.split()[0])
            pbmchip2name[line.split()[0]] = line.split()[1]

#revers the dictornary
name2pbmchip = {v: k for k, v in pbmchip2name.items()}


# In[45]:

sticky_tfs = pd.read_table("/home/ckibet/lustre/XGB-TFBSContext/Data/names.txt", header=None)
tf_list = []
for tf in sticky_tfs[0]:
    chip_list = glob.glob("/home/ckibet/lustre/Posneg/%s/*" % tf.capitalize())
    if len(chip_list) > 0:
        tf_list.append(tf)


# In[46]:

# ## Temp, they should be imported from MARS

# In[21]:

revcompl = lambda x: ''.join([{'A':'T','C':'G','G':'C','T':'A'}[B] for B in x][::-1])
def mkdir_p(path):
    import os
    import errno

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def get_unique(tf):
    import glob
    lests = glob.glob("/home/ckibet/lustre/PBM/%s/*" % tf.capitalize())
    new = []
    for i in lests:
        main = i.split("_v")[0]
        if main in new:
            continue
        else:
            new.append(main)
    return new

def find_kmers(string, kmer_size):
    kmers = []
    for i in range(0, len(string)-kmer_size+1):
        kmers.append(string[i:i+kmer_size])
    return kmers

def get_kmer_dict(kmerscore, kmer_name):
    """
    Try and make this generalized by checking if the first line has header or not. 
    
    Also, check the number of columns and determine which one contains the E-scores
    """

    test = pd.read_table(kmerscore, index_col="8-mer.1", usecols=["8-mer", "E-score"])

    scoredict  = test.to_dict()["E-score"]
    # with open(kmerscore) as kmers:
    #     print kmerscore
    #     for line in kmers:
    #         ke, rems, val = line.split("\t")
    #
    #         scoredict[ke] = val
    return scoredict, kmer_name

def get_kmer_dict_rev(kmerscore, kmer_name):
    test = pd.read_table(kmerscore, index_col="8-mer", usecols=["8-mer", "E-score"])
    test.fillna(0, inplace=True)
    test2 = pd.read_table(kmerscore, index_col="8-mer.1", usecols=["8-mer.1", "E-score"])
    test2.index.name = "8-mer"
    test2.fillna(0, inplace=True)
    combined = test.append(test2)
    combined_dict = combined.to_dict()["E-score"]

    return combined_dict, kmer_name

def energy_score_kmer(seq,kmerdict,revcompl):
    k_mers=find_kmers(seq,8)
    tot_score = 0
    for kmer in k_mers:
        if kmer in kmerdict:
            score=float(kmerdict[kmer])
        else:
            kmer2=revcompl(kmer)
            score=float(kmerdict[kmer2])
        tot_score+=score
    return tot_score


# In[35]:

# ## 1. Transform deBruijn sequences by scoring
# 
# The idea behind this approach is to use the dn-hg frequency counts as a measure of k-mer preference to be found in an open chromatin site. The human genome DNase sites frequency difference is used to transform eh deBruijn sequences. 
# 

mot_path="%s/Results/PBM_Reranked" % BASE_DIR


def run_assess(tf, mot_path="%s/Results/PBM_Reranked" % BASE_DIR):
    user_motif = "%s/%s/%s.meme" % (mot_path, tf.capitalize(), tf.capitalize())
    chip_list = glob.glob("/home/ckibet/lustre/Posneg/%s/*" % tf.capitalize())
    score.run_all(tf.lower(), 'energyscore', user_motif, chip_list, "%s/%s" % (mot_path, tf.capitalize())) 


def combine_meme(tf, path="%s/Results/PBM_Reranked" % BASE_DIR):
    """
    Within a directory, after a seed and wobble run, 
    combine all meme output into a single file. 
    """
    meme_out = "%s/%s/%s.meme" % (path,tf.capitalize(), tf.capitalize())
    if os.path.isfile(meme_out):
        os.remove(meme_out)
    meme_motifs = glob.glob("%s/%s/*meme" % (path,tf.capitalize()))
    
    mot = meme_motifs[0]
    with open(mot, 'r') as wr:
        with open(meme_out, 'w') as out_f:
            tests = wr.readlines()
            head = tests[:8]
            out_f.writelines(head)
            #tail = tests[8:]
            #out_f.writelines(tail)
    for mot in meme_motifs:
        with open(mot, 'r') as wr:
            with open(meme_out, 'a') as out_f:
                tests = wr.readlines()
                #head = tests[:8]
                #out_f.writelines(head)
                tail = tests[8:]
                out_f.writelines(tail)
        
def run_SnW(tf, mot="%s/Data/PBM_Reranked" % BASE_DIR):
    """
    """
    out_dir = "%s/%s" % ("%s/Data/PBM_Reranked2" % BASE_DIR,tf.capitalize())
    res_dir = "%s/%s" % (mot,tf.capitalize())
    mkdir_p(res_dir)
    for probe in get_unique(tf):
        probe_n =probe.split("/")[-1]
        v1 = "%s/%s_v%i_reranked_Hg_dn-less.txt" % (out_dir, probe_n,1)
        v2 = "%s/%s_v%i_reranked_Hg_dn-less.txt" % (out_dir, probe_n,2)
        get_ipython().system(u'perl {scripts_path}/seed_and_wobble_twoarray.pl {v1} {v2} 8 {scripts_path}/patterns_8of10.txt {scripts_path}/patterns_4x44k_all_8mer.txt {res_dir}/{probe_n}_reranked')
        get_ipython().system(u'python {scripts_path}/wobble2meme.py {res_dir}/{probe_n}_reranked_8mers_pwm_combined.txt {res_dir}/{probe_n}_reranked.uniprobe {res_dir}/{probe_n}_reranked.meme {probe_n}_reranked')
    combine_meme(tf,mot)
    run_assess(tf, mot)
    

def run_SnW_normal(tf):
    """
    Performs a normal Seed and wobble run
    """
    script = "%s/Scripts" % BASE_DIR
    out_dir = "%s/Data/PBM_Reranked/%s" % (BASE_DIR,tf.capitalize())
    res_dir = "%s/Results/PBM_Normal/%s" % (BASE_DIR,tf.capitalize())
    mkdir_p(res_dir)
    for probe in get_unique(tf):
        probe_n =probe.split("/")[-1]
        v1 = "%s_v%i_deBruijn.txt" % (probe,1)
        print v1
        v2 = "%s_v%i_deBruijn.txt" % (probe,2)
        get_ipython().system(u'perl {script}/seed_and_wobble_twoarray.pl {v1} {v2} 8 {script}/patterns_8of10.txt {script}/patterns_4x44k_all_8mer.txt {res_dir}/{probe_n}_deBruijn')
        get_ipython().system(u'python {script}/wobble2meme.py {res_dir}/{probe_n}_deBruijn_8mers_pwm_combined.txt {res_dir}/{probe_n}_deBruijn.uniprobe {res_dir}/{probe_n}_deBruijn.meme {probe_n}_deBruijn')
    #combine_meme(tf, path="%s/Results/PBM_Normal" % BASE_DIR)
    #run_assess(tf, mot_path="%s/Results/PBM_Normal" % BASE_DIR)


# In[53]:

def run_SnW_counts(tf, scalled):
    """
    Uses the DN-HG frequency counts transformed using minimum absolute from scikit learn.  
    """
    script = "%sScripts" % BASE_DIR
    out_dir = "%s/Data/PBM_Reranked/%s" % (BASE_DIR,tf.capitalize())
    res_dir = "%s/Results/PBM_PWM/%s" % (BASE_DIR,tf.capitalize())
    mkdir_p(res_dir)
    for probe in get_unique(tf):
        print probe
        probe_n =probe.split("/")[-1]
        v1 = "%s_v%i_deBruijn.txt" % (probe,1)
        print v1
        v2 = "%s_v%i_deBruijn.txt" % (probe,2)
        get_ipython().system(u'perl {script}/counts_seed_and_wobble_twoarray.pl {v1} {v2} 8 {script}/patterns_8of10.txt {script}/patterns_4x44k_all_8mer.txt {scalled} {res_dir}/{probe_n}_Counts')
        get_ipython().system(u'python {script}/wobble2meme.py {res_dir}/{probe_n}_deBruijn_8mers_pwm_combined.txt {res_dir}/{probe_n}_deBruijn.uniprobe {res_dir}/{probe_n}_deBruijn.meme {probe_n}_deBruijn')
    combine_meme(tf)
    run_assess(tf)


def run_rerank(tf):
    """
    Takes as input, TF name and per
    """
    out_dir = "%s/Data/PBM_Reranked/%s" % (BASE_DIR,tf.capitalize())
    res_dir = "%s/Results/PBM_PWM/%s" % (BASE_DIR,tf.capitalize())
    script = "%s/Scripts" % BASE_DIR
    mkdir_p(res_dir)
    for probe in get_unique(tf):
        print probe
        probe_n =probe.split("/")[-1]
        v1 = "%s_v%i_deBruijn.txt" % (probe,1)
        v2 = "%s_v%i_deBruijn.txt" % (probe,2)
        scalled ="/home/ckibet/lustre/XGB-TFBSContext/Data/hg_dn_backround_noise_minmax_escore.txt"
        os.system(u'perl %s/counts_seed_and_wobble_twoarray.pl %s %s 8 %s/patterns_8of10.txt %s/patterns_4x44k_all_8mer.txt %s %s/%s_Counts' % 
        (scripts_path,v1,v2,scripts_path,scripts_path, scalled,res_dir,probe_n))

        os.system(u'perl %s/rerank.pl %s %s/%s_Counts_8mers_pwm_combined.txt %s/%s_v1_reranked.txt' %
         (scripts_path,v1,res_dir,probe_n,res_dir,probe_n))
        os.system(u'perl %s/rerank.pl %s %s/%s_Counts_8mers_pwm_combined.txt %s/%s_v2_reranked.txt' %
         (scripts_path,v2,res_dir,probe_n,res_dir,probe_n))
        
        os.system(u'perl %s/seed_and_wobble_twoarray.pl %s/%s_v1_reranked.txt %s/%s_v2_reranked.txt 8 %s/patterns_8of10.txt %s/patterns_4x44k_all_8mer.txt %s/%s_Counts_reranked' % 
        (scripts_path,res_dir,probe_n,res_dir,probe_n,scripts_path,scripts_path,res_dir,probe_n))
        os.system(u'python %s/wobble2meme.py %s/%s_Counts_reranked_8mers_pwm_combined.txt %s/%s_Counts_reranked.uniprobe %s/%s_Counts_reranked.meme %s_Counts_reranked' % 
        (scripts_path,res_dir,probe_n,res_dir,probe_n,res_dir,probe_n,probe_n))
    combine_meme(tf, "%s/Results/PBM_PWM" % BASE_DIR)
    run_assess(tf, "%s/Results/PBM_PWM" % BASE_DIR)


# In[18]:

remain_list = [
 'Rxra',
 'Mafk',
 'Sp1']

# In[ ]:

for tf in tf_list: #:
    print tf
    #run_assess(tf, "%s/Results/PBM_PWM" % BASE_DIR)
    run_rerank(tf)
