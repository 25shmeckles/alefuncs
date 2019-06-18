### author:  alessio.marcozzi@gmail.com
### version: 2019_06
### licence: MIT
### requires Python >= 3.6


from Bio import pairwise2,  Entrez, SeqIO
from Bio.SubsMat import MatrixInfo as matlist
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML

import tensorflow as tf

from urllib.request import urlopen
from urllib.parse import urlparse

from subprocess import call, check_output, run

from pyensembl import EnsemblRelease

from bs4 import BeautifulSoup

from collections import OrderedDict, Set, Mapping, deque, Counter
from operator import itemgetter
from itertools import islice, chain
from threading import Thread
from numbers import Number


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema

import pandas as pd
import regex
import re

import datetime, math, sys, hashlib, pickle, time, random, string, json, glob, os, signal
import httplib2 as http

from urllib.request import urlopen
from pyliftover import LiftOver

from PIL import Image


def jitter(n, mu=0, sigma=0.1):
    '''Return a jittered version of n'''
    return n + np.random.normal(mu, sigma, 1)


class TimeoutError(Exception):
    '''
    Custom error for Timeout class.
    '''
    pass


class Timeout:
    '''
    A timeout handler with context manager.
    Based on UNIX signals.
    '''
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
        
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
        
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def random_walk(lenght):
    '''int => np.array
    Return a random walk path.
    '''
    walk = []
    y = 0
    for _ in range(lenght):
        if random.randint(0,1):
            y += 1
        else:
            y -= 1
        walk.append(y)
    return np.array(walk)


def find_min_max(array):
    '''np.array => dict
    Return a dictionary of indexes
    where the maxima and minima of the input array are found.
    '''
    # for local maxima
    maxima = argrelextrema(array, np.greater)

    # for local minima
    minima = argrelextrema(array, np.less)
    
    return {'maxima':maxima,
            'minima':minima}


def smooth(array, window_len=10, window='hanning'):
    '''np.array, int, str => np.array
    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
        t = linspace(-2,2,0.1)
        x = sin(t)+randn(len(t))*0.1
        y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    '''

    if array.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if array.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[array[window_len-1:0:-1],array,array[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    
    y = y[int(window_len/2-1):-int(window_len/2)]
    offset = len(y)-len(array) #in case input and output are not of the same lenght
    assert len(array) == len(y[offset:])
    return y[offset:] 



def cohen_effect_size(group1, group2):
    '''(np.array, np.array) => float
    Compute the Cohen Effect Size (d) between two groups
    by comparing the difference between groups to the variability within groups.
    Return the the difference in standard deviation.
    '''
    assert type(group1) == np.ndarray
    assert type(group2) == np.ndarray

    diff = group1.mean() - group2.mean()
    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d



def gen_ascii_symbols(input_file, chars):
    '''
    Return a dict of letters/numbers associated with
    the corresponding ascii-art representation.
    You can use http://www.network-science.de/ascii/ to generate the ascii-art for each symbol.
        
    The input file looks like:
    ,adPPYYba,  
    ""     `Y8  
    ,adPPPPP88  
    88,    ,88  
    `"8bbdP"Y8  
    88           
    88           
    88           
    88,dPPYba,   
    88P'    "8a  
    88       d8  
    88b,   ,a8"  
    8Y"Ybbd8"'
    
    ...
    
    Each symbol is separated by at least one empty line ("\n")
    '''
    #input_file = 'ascii_symbols.txt'
    #chars = string.ascii_lowercase+string.ascii_uppercase+'0123456789'

    symbols = []
    s = ''
    with open(input_file, 'r') as f:
        for line in f:
            if line == '\n':
                if len(s):
                    symbols.append(s)
                    s = ''
                else:
                    continue
            else:
                s += line

    return dict(zip(chars,symbols))



def gen_ascii_captcha(symbols, length=6, max_h=10, noise_level=0, noise_char='.'):
    '''
    Return a string of the specified length made by random symbols.
    Print the ascii-art representation of it.
    
    Example:
    
    symbols = gen_ascii_symbols(input_file='ascii_symbols.txt',
                                chars = string.ascii_lowercase+string.ascii_uppercase+'0123456789')
    while True:
        captcha = gen_ascii_captcha(symbols, noise_level=0.2)
        x = input('captcha: ')
        if x == captcha:
            print('\ncorrect')
            break
        print('\ninvalid captcha, please retry')
    '''
    assert noise_level <= 1
    #max_h = 10
    #noise_level = 0
    captcha = ''.join(random.sample(chars, length))
    #print(code)
    pool = [symbols[c].split('\n') for c in captcha]

    for n in range(max_h, 0, -1):
        line = ''
        for item in pool:
            try:
                next_line = item[-n]
            except IndexError:
                next_line = ''.join([' ' for i in range(max([len(_item) for _item in item]))])

            if noise_level:
                #if random.random() < noise_level:
                #    next_line = next_line.replace(' ', noise_char)
                next_line = ''.join([c if random.random() > noise_level \
                                     else random.choice(noise_char) for c in next_line])
            line += next_line

        print(line)
    return captcha


def rnd_sample_df(df, n=1, slice_size=1):
    '''
    Yield dataframes generated by randomly slicing df.
    It is different from pandas.DataFrame.sample().
    '''
    assert n > 0 and slice_size > 0
    max_len = len(df)-slice_size
    for _ in range(n):
        i = random.randint(0,max_len)
        yield df.iloc[i:i+slice_size] 


def date_to_stamp(d='2012-12-31'):
    '''
    Return UNIX timestamp of a date.
    '''
    Y,M,D = d.split('-')
    stamp = time.mktime(datetime.date(int(Y),
                                      int(M),
                                      int(D)
                                     ).timetuple()
                       )
    return stamp


def rolling_normalize_df(df, method='min-max', size=30, overlap=5):
    '''
    Return a new df with datapoints normalized based on a sliding window
    of rolling on the a pandas.DataFrame.
    It is useful to have local (window by window) normalization of the values.
    '''
    
    to_merge = []
    for item in split_overlap_long(df, size, overlap, is_dataframe=True):
        to_merge.append(normalize_df(item, method))
                
    new_df = pd.concat(to_merge)
    return new_df.groupby(new_df.index).mean()


def normalize_df(df, method='min-max'):
    '''
    Return normalized data.
    max, min, mean and std are computed considering
    all the values of the dfand not by column.
    i.e. mean = df.values.mean() and not df.mean().
    Ideal to normalize df having multiple columns of non-indipendent values.
    Methods implemented:
                       'raw'      No normalization
                       'min-max'  Deafault
                       'norm'     ...
                       'z-norm'   ...
                       'sigmoid'  ...
                       'decimal'  ...
                       'softmax'  It's a transformation rather than a normalization
                       'tanh'     ...
    '''
    if type(df) is not pd.core.frame.DataFrame:
        df = pd.DataFrame(df)
        
    if method == 'min-max':
        return (df-df.values.min())/(df.values.max()-df.values.min())

    if method == 'norm':
        return (df-df.values.mean())/(df.values.max()-df.values.mean())

    if method == 'z-norm':
        return (df-df.values.mean())/df.values.std()

    if method == 'sigmoid':
        _max = df.values.max()
        return df.apply(lambda x: 1/(1+np.exp(-x/_max)))

    if method == 'decimal':
        #j = len(str(int(df.values.max())))
        i = 10**len(str(int(df.values.max())))#10**j
        return df.apply(lambda x: x/i)

    if method == 'tanh':
        return 0.5*(np.tanh(0.01*(df-df.values.mean()))/df.values.std() + 1)

    if method == 'softmax':
        return np.exp(df)/np.sum(np.exp(df))

    if method == 'raw':
        return df

    raise ValueError(f'"method" not found: {method}')



def merge_dict(dictA, dictB):
    '''(dict, dict) => dict
    Merge two dicts, if they contain the same keys, it sums their values.
    Return the merged dict.
    
    Example:
        dictA = {'any key':1, 'point':{'x':2, 'y':3}, 'something':'aaaa'}
        dictB = {'any key':1, 'point':{'x':2, 'y':3, 'z':0, 'even more nested':{'w':99}}, 'extra':8}
        merge_dict(dictA, dictB)
        
        {'any key': 2,
         'point': {'x': 4, 'y': 6, 'z': 0, 'even more nested': {'w': 99}},
         'something': 'aaaa',
         'extra': 8}
    '''
    r = {}
    
    common_k = [k for k in dictA if k in dictB]
    common_k += [k for k in dictB if k in dictA]
    common_k = set(common_k)
    
    for k, v in dictA.items():
        #add unique k of dictA
        if k not in common_k:
            r[k] = v
        
        else:
            #add inner keys if they are not containing other dicts 
            if type(v) is not dict:
                if k in dictB:
                    r[k] = v + dictB[k]
            else:
                #recursively merge the inner dicts
                r[k] = merge_dict(dictA[k], dictB[k])
    
    #add unique k of dictB
    for k, v in dictB.items():
        if k not in common_k:
            r[k] = v

    return r


def png_to_flat_array(img_file):
    img = Image.open(img_file).convert('RGBA')
    arr = np.array(img)
    # make a 1-dimensional view of arr
    return arr.ravel()


def png_to_vector_matrix(img_file):
    # convert it to a matrix
    return np.matrix(png_to_flat_array(img_file))



def TFKMeansCluster(vectors, noofclusters, datatype="uint8"):
    '''
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    '''
 
    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)
 
    #Find out the dimensionality
    dim = len(vectors[0])
 
    #Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    random.shuffle(vector_indices)
 
    #GRAPH OF COMPUTATION
    #We initialize a new graph and set it as the default during each run
    #of this algorithm. This ensures that as this function is called
    #multiple times, the default graph doesn't keep getting crowded with
    #unused ops and Variables from previous function calls.
 
    graph = tf.Graph()
 
    with graph.as_default():
 
        #SESSION OF COMPUTATION
 
        sess = tf.Session()
 
        ##CONSTRUCTING THE ELEMENTS OF COMPUTATION
 
        ##First lets ensure we have a Variable vector for each centroid,
        ##initialized to one of the vectors from the available data points
        centroids = [tf.Variable((vectors[vector_indices[i]])) for i in range(noofclusters)]
        
        ##These nodes will assign the centroid Variables the appropriate
        ##values
        centroid_value = tf.placeholder(datatype, [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))
 
        ##Variables for cluster assignments of individual vectors(initialized
        ##to 0 at first)
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        ##These nodes will assign an assignment Variable the appropriate
        ##value
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))
 
        ##Now lets construct the node that will compute the mean
        #The placeholder for the input
        mean_input = tf.placeholder("float", [None, dim])
        #The Node/op takes the input and computes a mean along the 0th
        #dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)
 
        ##Node for computing Euclidean distances
        #Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))
 
        ##This node will figure out which cluster to assign a vector to,
        ##based on Euclidean distances of the vector from the centroids.
        #Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)
 
        ##INITIALIZING STATE VARIABLES
 
        ##This will help initialization of all Variables defined with respect
        ##to the graph. The Variable-initializer should be defined after
        ##all the Variables have been constructed, so that each of them
        ##will be included in the initialization.
        init_op = tf.global_variables_initializer() #deprecated tf.initialize_all_variables()
        
 
        #Initialize all variables
        sess.run(init_op)
 
        ##CLUSTERING ITERATIONS
 
        #Now perform the Expectation-Maximization steps of K-Means clustering
        #iterations. To keep things simple, we will only do a set number of
        #iterations, instead of using a Stopping Criterion.
        noofiterations = 100
        for iteration_n in range(noofiterations):
 
            ##EXPECTATION STEP
            ##Based on the centroid locations till last iteration, compute
            ##the _expected_ centroid assignments.
            #Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                #Compute Euclidean distance between this vector and each
                #centroid. Remember that this list cannot be named
                #'centroid_distances', since that is the input to the
                #cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                #Now use the cluster assignment node, with the distances
                #as the input
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                #Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})
 
            ##MAXIMIZATION STEP
            #Based on the expected state computed from the Expectation Step,
            #compute the locations of the centroids so as to maximize the
            #overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                #Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                #Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={mean_input: np.array(assigned_vects)})
                #Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={centroid_value: new_location})
 
        #Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments

def xna_calc(sequence, t='dsDNA', p=0):
    '''str => dict
    BETA version, works only for dsDNA and ssDNA.
    Return basic "biomath" calculations based on the input sequence.
    Arguments:
        t (type) :'ssDNA' or 'dsDNA'
        p (phosphates): 0,1,2
        #in case if ssDNA having 3'P, you should pass 2 i.e., 2 phospates present in 1 dsDNA molecule 
    '''
    r = {}
    
    #check inputs
    c = Counter(sequence.upper())
    for k in c.keys():
        if k in 'ACGNT':
            pass
        else:
            raise ValueError(f'Wrong sequence passed: "sequence" contains invalid characters, only "ATCGN" are allowed.')
    if t not in ['ssDNA','dsDNA']:
        raise ValueError(f'Wrong DNA type passed: "t" can be "ssDNA" or "dsDNA". "{t}" was passed instead.')
    if not 0 <= p <= 2:
        raise ValueError(f'Wrong number of 5\'-phosphates passed: "p" must be an integer from 0 to 4. {p} was passed instead.')
    
    
    ##Calculate:
    
    #length
    r['len'] = len(sequence)
    

    #molecular weight
    #still unsure about what is the best method to do this
    
    #s = 'ACTGACTGACTATATTCGCGATCGATGCGCTAGCTCGTACGC'
    #bioinformatics.org : 25986.8  Da
    #Thermo             : 25854.8  Da 
    #Promega            : 27720.0  Da 
    #MolBioTools        : 25828.77 Da
    #This function      : 25828.86 Da #Similar to OligoCalc implementation
    
    #DNA Molecular Weight (typically for synthesized DNA oligonucleotides.
    #The OligoCalc DNA MW calculations assume that there is not a 5' monophosphate)
    #Anhydrous Molecular Weight = (An x 313.21) + (Tn x 304.2) + (Cn x 289.18) + (Gn x 329.21) - 61.96
    #An, Tn, Cn, and Gn are the number of each respective nucleotide within the polynucleotide.
    #The subtraction of 61.96 gm/mole from the oligonucleotide molecular weight takes into account the removal
    #of HPO2 (63.98) and the addition of two hydrogens (2.02).
    #Alternatively, you could think of this of the removal of a phosphate and the addition of a hydroxyl,
    #since this formula calculates the molecular weight of 5' and 3' hydroxylated oligonucleotides.
    #Please note: this calculation works well for synthesized oligonucleotides.
    #If you would like an accurate MW for restriction enzyme cut DNA, please use:
    #Molecular Weight = (An x 313.21) + (Tn x 304.2) + (Cn x 289.18) + (Gn x 329.21) - 61.96 + 79.0
    #The addition of 79.0 gm/mole to the oligonucleotide molecular weight takes into account the 5' monophosphate
    #left by most restriction enzymes.
    #No phosphate is present at the 5' end of strands made by primer extension,
    #so no adjustment to the OligoCalc DNA MW calculation is necessary for primer extensions.
    #That means that for ssDNA, you need to add 79.0 to the value calculated by OligoCalc
    #to get the weight with a 5' monophosphate.
    #Finally, if you need to calculate the molecular weight of phosphorylated dsDNA,
    #don't forget to adjust both strands. You can automatically perform either addition
    #by selecting the Phosphorylated option from the 5' modification select list.
    #Please note that the chemical modifications are only valid for DNA and may not be valid for RNA
    #due to differences in the linkage chemistry, and also due to the lack of the 5' phosphates
    #from synthetic RNA molecules. RNA Molecular Weight (for instance from an RNA transcript).
    #The OligoCalc RNA MW calculations assume that there is a 5' triphosphate on the molecule)
    #Molecular Weight = (An x 329.21) + (Un x 306.17) + (Cn x 305.18) + (Gn x 345.21) + 159.0
    #An, Un, Cn, and Gn are the number of each respective nucleotide within the polynucleotide.
    #Addition of 159.0 gm/mole to the molecular weight takes into account the 5' triphosphate.
    
    if t == 'ssDNA':
        mw = ((c['A']*313.21)+(c['T']*304.2)+(c['C']*289.18)+(c['G']*329.21)+(c['N']*303.7)-61.96)+(p*79.0)
        
    elif t =='dsDNA':
        mw_F = ((c['A']*313.21)+(c['T']*304.2)+(c['C']*289.18)+(c['G']*329.21)+(c['N']*303.7)-61.96)+(p*79.0)
        d = Counter(complement(sequence.upper())) #complement sequence
        mw_R = ((d['A']*313.21)+(d['T']*304.2)+(d['C']*289.18)+(d['G']*329.21)+(d['N']*303.7)-61.96)+(p*79.0)
        mw = mw_F + mw_R
    elif t == 'ssRNA':
        pass
    elif t == 'dsRNA':
        pass
    else:
        return ValueError(f'Nucleic acid type not understood: "{t}"')
        
    r['MW in Daltons'] = mw
    
    #in ng
    r['MW in ng'] = mw * 1.6605402e-15
    
    #molecules in 1ng
    r['molecules per ng'] = 1/r['MW in ng']
    
    #ng for 10e10 molecules
    r['ng per billion molecules'] = (10**9)/r['molecules per ng'] #(1 billions)
    
    #moles per ng
    r['moles per ng'] = (r['MW in ng'] * mw)
    return r


def occur(string, sub):
    '''
    Counts the occurrences of a sequence in a string considering overlaps.
    Example:
            >> s = 'ACTGGGACGGGGGG'
            >> s.count('GGG')
            3
            >> occur(s,'GGG')
            5
    '''
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count


def get_prime(n):
    for num in range(2,n,2):
        if all(num%i != 0 for i in range(2,int(math.sqrt(num))+1)):
            yield num

            
def ssl_fencrypt(infile, outfile):
    '''(file_path, file_path) => encrypted_file
    Uses openssl to encrypt/decrypt files.
    '''
    pwd = getpass('enter encryption pwd:')
    if getpass('repeat pwd:') == pwd:
        run(f'openssl enc -aes-256-cbc -a -salt -pass pass:{pwd} -in {infile} -out {outfile}',shell=True)
    else:
        print("passwords don't match.")

    
def ssl_fdecrypt(infile, outfile):
    '''(file_path, file_path) => decrypted_file
    Uses openssl to encrypt/decrypt files.
    '''
    pwd = getpass('enter decryption pwd:')
    run(f'openssl enc -d -aes-256-cbc -a -pass pass:{pwd} -in {infile} -out {outfile}', shell=True)     

    
def loop_zip(strA, strB):
    '''(str, str) => zip()
    Return a zip object containing each letters of strA, paired with letters of strB.
    If strA is longer than strB, then its letters will be paired recursively.
    Example:
        >>> list(loop_zip('ABCDEF', '123'))
        [('A', '1'), ('B', '2'), ('C', '3'), ('D', '1'), ('E', '2'), ('F', '3')]
    '''
    assert len(strA) >= len(strB)
    s = ''
    n = 0
    for l in strA:
        try:
            s += strB[n]
        except IndexError:
            n = 0
            s += strB[n]
        n += 1
    return zip(list(strA),list(s))


def encrypt(msg, pwd):
    '''(str, str) => list
    Simple encryption/decription tool.
    WARNING:
    This is NOT cryptographically secure!!
    '''
    if len(msg) < len(pwd):
        raise ValueError('The password is longer than the message. This is not allowed.')
    return [(string_to_number(a)+string_to_number(b)) for a,b in loop_zip(msg, pwd)]


def decrypt(encr, pwd):
    '''(str, str) => list
    Simple encryption/decription tool.
    WARNING:
    This is NOT cryptographically secure!!
    '''
    return ''.join([number_to_string((a-string_to_number(b))) for a,b in loop_zip(encr, pwd)])


def convert_mw(mw, to='g'):
    '''(int_or_float, str) => float
    Converts molecular weights (in dalton) to g, mg, ug, ng, pg.
    Example:
            >> diploid_human_genome_mw = 6_469.66e6 * 660 #lenght * average weight of nucleotide
            >> convert_mw(diploid_human_genome_mw, to="ng")
            0.0070904661368191195
    '''
    if to == 'g':
        return mw * 1.6605402e-24
    if to == 'mg':
        return mw * 1.6605402e-21
    if to == 'ug':
        return mw * 1.6605402e-18
    if to == 'ng':
        return mw * 1.6605402e-15
    if to == 'pg':
        return mw * 1.6605402e-12
    raise ValueError(f"'to' must be one of ['g','mg','ug','ng','pg'] but '{to}' was passed instead.")


def snp237(snp_number):
    '''int => list
    Return the genomic position of a SNP on the GCRh37 reference genome.
    '''
    query = f'https://www.snpedia.com/index.php/Rs{snp_number}'
    html = urlopen(query).read().decode("utf-8")
    for line in html.split('\n'):
        if line.startswith('<tr><td width="90">Reference</td>'):
            reference = line.split('"')[-2]
        elif line.startswith('<tr><td width="90">Chromosome</td>'):
            chromosome = line.split('<td>')[1].split('<')[0]
        elif line.startswith('<tr><td width="90">Position</td>'):
            position = int(line.split('<td>')[1].split('<')[0])
            break

    if 'GRCh38' in reference:
        lo = LiftOver('hg38', 'hg19')
        return lo.convert_coordinate(f'chr{chromosome}', position)[0][:2]
    else:
        return f'chr{chromosome}', position


def is_prime(n):
    '''Return True if n is a prime number'''

    if n == 1: 
        return False #1 is not prime

    #if it's even and not 2, then it's not prime
    if n == 2:
        return True
    if n > 2 and n % 2 == 0:
        return False

    max_divisor = math.floor(math.sqrt(n))
    for d in range(3, 1 + max_divisor, 2):
        if n % d == 0:
            return False
    return True



def flatmap(f, items):
    return chain.from_iterable(imap(f, items))


def parse_fasta(fasta_file):
    '''file_path => dict
    Return a dict of id:sequences.
    '''
    d = {}
    _id = False
    seq = ''
    with open(fasta_file,'r') as f:
        for line in f:
            if line.startswith('\n'):
                continue
            if line.startswith('>'):
                if not _id:
                    _id = line[1:].strip()
                elif _id and seq:
                    d.update({_id:seq})
                    _id = line[1:].strip()
                    seq = ''
            else:
                seq += line.strip()
        d.update({_id:seq})
    return d


def get_fasta_stats(fasta_file):
    '''file_path => dict
    Return lenght and base counts of each seuqence found in the fasta file.
    '''
    d = {}
    _id = False
    seq = ''
    with open(fasta_file,'r') as f:
        for line in f:
            if line.startswith('\n'):
                continue
            if line.startswith('>'):
                if not _id:
                    _id = line[1:].strip()
                elif _id and seq:
                    d.update({_id:seq})
                    _id = line[1:].strip()
                    seq = ''
            else:
                seq += line.strip().upper()
        d.update({_id:{'length':len(seq),
                       'A':seq.count('A'),
                       'T':seq.count('T'),
                       'C':seq.count('C'),
                       'G':seq.count('G'),
                       'N':seq.count('N')}
                 })
    return d


def quick_align(reference, sample, matrix=matlist.blosum62, gap_open=-10, gap_extend=-0.5):
    '''
    Return a binary score matrix for a pairwise alignment.
    '''

    alns = pairwise2.align.globalds(reference, sample, matrix, gap_open, gap_extend)

    top_aln = alns[0]
    aln_reference, aln_sample, score, begin, end = top_aln

    score = []
    for i, base in enumerate(aln_reference):
        if aln_sample[i] == base:
            score.append(1)
        else:
            score.append(0)

    return score


def vp(var_name,var_dict=globals(),sep=' : '):
    '''(str, dict) => print
    Variable Print, a fast way to print out a variable's value.
    >>> scale = 0.35
    >>> mass = '71 Kg'
    >>> vp('scale')
    scale : 0.35
    >>> vp('mass',sep='=')
    mass=71 Kg
    '''
    try:
        print(f'{var_name}{sep}{g[var_name]}')
    except:
        print(f'{var_name} not found!')


def view_matrix(arrays):
    '''list_of_arrays => print
    Print out the array, row by row.
    '''
    for a in arrays:
        print(a)
    print('=========')
    for n,r in enumerate(arrays):
        print(n,len(r))
    print(f'row:{len(arrays)}\ncol:{len(r)}')


def fill_matrix(arrays,z=0):
    '''(list_of_arrays, any) => None
    Add z to fill-in any array shorter than m=max([len(a) for a in arrays]).
    '''
    m = max([len(a) for a in arrays])
    for i,a in enumerate(arrays):
        if len(a) != m:
            arrays[i] = np.append(a, [z for n in range(m-len(a))])


def get_size(obj_0):
    '''obj => int
    Recursively iterate to sum size of object & members (in bytes).
    Adapted from http://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    '''
    def inner(obj, _seen_ids = set()):
        zero_depth_bases = (str, bytes, Number, range, bytearray)
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def total_size(o, handlers={}, verbose=False):
    '''(object, dict, bool) => print
    Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
                    
    >>> d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
    >>> print(total_size(d, verbose=True))
    
        796
        280 <type 'dict'> {'a': 1, 'c': 3, 'b': 2, 'e': 'a string of chars', 'd': [4, 5, 6, 7]}
        38 <type 'str'> 'a'
        24 <type 'int'> 1
        38 <type 'str'> 'c'
        24 <type 'int'> 3
        38 <type 'str'> 'b'
        24 <type 'int'> 2
        38 <type 'str'> 'e'
        54 <type 'str'> 'a string of chars'
        38 <type 'str'> 'd'
        104 <type 'list'> [4, 5, 6, 7]
        24 <type 'int'> 4
        24 <type 'int'> 5
        24 <type 'int'> 6
        24 <type 'int'> 7
    '''
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)   # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s,type(o),repr(o))

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def center(pattern):
    '''np.array => np.array
    Return the centered pattern, 
    which is given by [(value - mean) for value in pattern]
    >>> array = np.array([681.7, 682.489, 681.31, 682.001, 682.001, 682.499, 682.001])
    >>> center(array)
    array([-0.30014286,  0.48885714, -0.69014286,  0.00085714,  0.00085714, 0.49885714,  0.00085714])
    '''
    #mean = pattern.mean()
    #return np.array([(value - mean) for value in pattern])
    return (pattern - np.mean(pattern))


def rescale(pattern):
    '''np.array => np.array
    Rescale each point of the array to be a float between 0 and 1.
    >>> a =  np.array([1,2,3,4,5,6,5,4,3,2,1])
    >>> rescale(a)
    array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ,  0.8,  0.6,  0.4,  0.2,  0. ])
    '''
    #_max = pattern.max()
    #_min = pattern.min()
    #return np.array([(value - _min)/(_max - _min) for value in pattern])
    return (pattern - pattern.min()) / (pattern.max()-pattern.min())


def standardize(pattern):
    '''np.array => np.array
    Return a standard pattern.
    >>> a =  np.array([1,2,3,4,5,6,5,4,3,2,1])
    >>> standardize(a)
    array([-1.41990459, -0.79514657, -0.17038855,  0.45436947,  1.07912749,
            1.7038855 ,  1.07912749,  0.45436947, -0.17038855, -0.79514657,
           -1.41990459])
    '''
    #mean = pattern.mean()
    #std = pattern.std()
    #return np.array([(value - mean)/std for value in pattern])
    return (pattern - np.mean(pattern)) / np.std(pattern)


def normalize(pattern):
    '''np.array => np.array
    Return a normalized pattern using np.linalg.norm().
    >>> a =  np.array([1,2,3,4,5,6,5,4,3,2,1])
    >>> normalize(a)
    '''
    return pattern / np.linalg.norm(pattern)


def gen_patterns(data, length, ptype='all'):
    '''(array, int) => dict
    Generate all possible patterns of a given legth
    by manipulating consecutive slices of data.
    Return a dict of patterns dividad by pattern_type.
    >>> data = [1,2,3,4,5,4,3,2,1]
    >>> gen_patterns(data,len(data))
    {'center': {0: array([-1.77777778, -0.77777778,  0.22222222,  1.22222222,  2.22222222,  1.22222222,  0.22222222, -0.77777778, -1.77777778])},
       'norm': {0: array([ 0.10846523,  0.21693046,  0.32539569,  0.43386092,  0.54232614,  0.43386092,  0.32539569,  0.21693046,  0.10846523])},
      'scale': {0: array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  0.75,  0.5 ,  0.25,  0.  ])},
        'std': {0: array([-1.35224681, -0.59160798,  0.16903085,  0.92966968,  1.69030851,  0.92966968,  0.16903085, -0.59160798, -1.35224681])}}
    >>> gen_patterns(data,3)
    {'center': {0: array([-1.,  0.,  1.]),
                1: array([-1.,  0.,  1.]),
                2: array([-1.,  0.,  1.])},
       'norm': {0: array([ 0.26726124,  0.53452248,  0.80178373]),
                1: array([ 0.37139068,  0.55708601,  0.74278135]),
                2: array([ 0.42426407,  0.56568542,  0.70710678])},
      'scale': {0: array([ 0. ,  0.5,  1. ]),
                1: array([ 0. ,  0.5,  1. ]),
                2: array([ 0. ,  0.5,  1. ])},
        'std': {0: array([-1.22474487,  0.        ,  1.22474487]),
                1: array([-1.22474487,  0.        ,  1.22474487]),
                2: array([-1.22474487,  0.        ,  1.22474487])}}
    '''

    results = {}
    ptypes = ['std','norm','scale','center']
    if ptype == 'all': #to do: select specific ptypes
        for t in ptypes:
            results.update({t:{}})

        for n in range(length):
            if n+length > len(data):
                break

            raw = np.array(data[n:n+length])

            partial = {'std'   :standardize(raw),
                       'norm'  :normalize(raw),
                       'scale' :rescale(raw),
                       'center':center(raw)}

            for t in ptypes:
                results[t].update({n:partial[t]})
        return results

def delta_percent(a, b, warnings=False):
    '''(float, float) => float
    Return the difference in percentage between a nd b. 
    If the result is 0.0 return 1e-09 instead.
    >>> delta_percent(20,22)
    10.0
    >>> delta_percent(2,20)
    900.0
    >>> delta_percent(1,1)
    1e-09
    >>> delta_percent(10,9)
    -10.0
    '''
    #np.seterr(divide='ignore', invalid='ignore')
    try:
        x = ((float(b)-a) / abs(a))*100
        if x == 0.0:
            return 0.000000001 #avoid -inf
        else:
            return x
    except Exception as e:
        if warnings:
            print(f'Exception raised by delta_percent(): {e}')
        return 0.000000001 #avoid -inf


def is_similar(array1,array2,t=0.1):
    '''(array, array, float) => bool
    Return True if all the points of two arrays are no more than t apart.
    '''
    if len(array1) != len(array2):
        return False
    for i,n in enumerate(array1):
        if abs(n-array2[i]) <= t:
            pass
        else:
            return False
    return True


def cluster_patterns(pattern_list,t):
    ''' ([array, array, ...], float) => dict
    Return a dict having as keys the idx of patterns in pattern_list 
    and as values the idx of the similar patterns.
    "t" is the inverse of a similarity threshold, 
    i.e. the max discrepancy between the value of array1[i] and array2[i].
    If no simalar patterns are found,value is assigned to an empty list.
    >>> a  = [1,2,3,4,5,6,5,4,3,2,1]
    >>> a1 = [n+1 for n in a]
    >>> a2 = [n+5 for n in a]
    >>> a3 = [n+6 for n in a]
    >>> patterns = [a,a1,a2,a3]
    >>> cluster_patterns(patterns,t=2)
    {0: [1], 1: [0], 2: [3], 3: [2]}
    >>> cluster_patterns(patterns,t=5)
    {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
    >>> cluster_patterns(patterns,t=0.2)
    {0: [], 1: [], 2: [], 3: []}
    '''
    result = {}
    for idx, array1 in enumerate(pattern_list):
        result.update({idx:[]})
        for i,array2 in enumerate(pattern_list):
            if i != idx:
                if is_similar(array1,array2,t=t):
                    result[idx].append(i)
    #print 'clusters:',len([k for k,v in result.iteritems() if len(v)])
    return result


def stamp_to_date(stamp,time='utc'):
    '''(int_or_float, float, str) => datetime.datetime
    Convert UNIX timestamp to UTC or Local Time
    >>> stamp = 1477558868.93
    >>> print stamp_to_date(stamp,time='utc')
    2016-10-27 09:01:08.930000
    >>> print stamp_to_date(int(stamp),time='utc')
    2016-10-27 09:01:08
    >>> stamp_to_date(stamp,time='local')
    datetime.datetime(2016, 10, 27, 11, 1, 8, 930000)
    '''

    if time.lower() == 'utc':
        return datetime.datetime.utcfromtimestamp(stamp)
    elif time.lower() == 'local':
        return datetime.datetime.fromtimestamp(stamp)
    else:
        raise ValueError('"time" must be "utc" or "local"')


def future_value(interest,period,cash):
    '''(float, int, int_or_float) => float
    Return the future value obtained from an amount of cash
    growing with a fix interest over a period of time.
    >>> future_value(0.5,1,1)
    1.5
    >>> future_value(0.1,10,100)
    259.37424601
    '''
    if not 0 <= interest <= 1:
        raise ValueError('"interest" must be a float between 0 and 1')

    for d in range(period):
        cash += cash * interest
    return cash


def entropy(sequence, verbose=False):
    '''(string, bool) => float
    Return the Shannon Entropy of a string.
    Calculated as the minimum average number of
    bits per symbol required for encoding the string.
    The theoretical limit for data compression:
    Shannon Entropy of the string * string length
    '''
    letters = list(sequence)
    alphabet = list(set(letters)) # list of symbols in the string
    # calculate the frequency of each symbol in the string
    frequencies = []
    for symbol in alphabet:
        ctr = 0
        for sym in letters:
            if sym == symbol:
                ctr += 1
        frequencies.append(float(ctr) / len(letters))
    
    # Shannon entropy
    ent = 0.0
    for freq in frequencies:
        ent = ent + freq * math.log(freq, 2)
    ent = -ent
    if verbose:
        print('Input string:')
        print(sequence)
        print()
        print('Alphabet of symbols in the string:')
        print(alphabet)
        print()
        print('Frequencies of alphabet symbols:')
        print(frequencies)
        print()
        print('Shannon entropy:')
        print(ent)
        print('Minimum number of bits required to encode each symbol:')
        print(int(math.ceil(ent)))
        
    return ent


def quick_entropy(sequence):
    '''(string, bool) => float
    Return the Shannon Entropy of a string.
    Compact version of entropy()
    Calculated as the minimum average number of bits per symbol
    required for encoding the string.
    The theoretical limit for data compression:
    Shannon Entropy of the string * string length.
    '''
    
    alphabet = set(sequence) # list of symbols in the string

    # calculate the frequency of each symbol in the string
    frequencies = []
    for symbol in alphabet:
        frequencies.append(sequence.count(symbol) / len(sequence))

    # Shannon entropy
    ent = 0.0
    for freq in frequencies:
        ent -= freq * math.log(freq, 2)
        
    return ent


def percent_of(total, fraction):
    '''(int_or_float,int_or_float) => float
    Return the percentage of 'fraction' in 'total'.
    
    Examples:
        percent_of(150, 75)
        >>> 50.0
        
        percent_of(30, 90)
        >>> 300.0
    '''
    assert total > 0
    if np.isnan(total) or np.isnan(fraction):
        return nan
    return (100*fraction)/total


def buzz(sequence, noise=0.01):
    '''(string,float) => string
    Return a sequence with some random noise.
    '''
    if not noise:
        return sequence
    bits = set([char for char in sequence] + ['del','dup'])
    r = ''
    for char in sequence:
        if random.random() <= noise:
            b = random.sample(bits,1)[0]
            if b == 'del':
                continue
            elif b == 'dup':
                r += 2*char
            else:
                r += b
        else:
            r += char
    return r


def simple_consensus(aligned_sequences_file):
    '''file => string
    Return the consensus of a series of fasta sequences aligned with muscle.
    '''
    # Generate consensus from Muscle alignment
    sequences = []
    seq = False
    with open(aligned_sequences_file,'r') as f:
        for line in f:
            if line.startswith('\n'):
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                seq = ''
            else:
                seq += line.strip()
        sequences.append(seq)
    #check if all sequenced have the same length
    for seq in sequences:
        assert len(seq) == len(sequences[0])
    
    #compute consensus by majority vote
    consensus = ''
    for i in range(len(sequences[0])):
        char_count = Counter()
        for seq in sequences:
            char_count.update(seq[i])
        consensus += char_count.most_common()[0][0]

    return consensus.replace('-','')


def print_sbar(n,m,s='|#.|',size=30,message=''):
    '''(int,int,string,int) => None
    Print a progress bar using the simbols in 's'.
    Example:
        range_limit = 1000
        for n in range(range_limit):
            print_sbar(n+1,m=range_limit)
            time.sleep(0.1)
    '''
    #adjust to bar size
    if m != size:
        n =(n*size)/m
        m = size
    #calculate ticks
    _a = int(n)*s[1]+(int(m)-int(n))*s[2]
    _b = round(n/(int(m))*100,1)
    #adjust overflow
    if _b >= 100:
        _b = 100.0
    #to stdout    
    sys.stdout.write(f'\r{message}{s[0]}{_a}{s[3]} {_b}%     ')
    sys.stdout.flush()


def get_hash(a_string,algorithm='md5'):
    '''str => str
    Return the hash of a string calculated using various algorithms.
    
    .. code-block:: python
        >>> get_hash('prova','md5')
        '189bbbb00c5f1fb7fba9ad9285f193d1'
        >>> get_hash('prova','sha256')
        '6258a5e0eb772911d4f92be5b5db0e14511edbe01d1d0ddd1d5a2cb9db9a56ba'
    '''
    if algorithm == 'md5':
        return hashlib.md5(a_string.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(a_string.encode()).hexdigest()
    else:
        raise ValueError('algorithm {} not found'.format(algorithm))


def get_first_transcript_by_gene_name(gene_name):
    '''str => str
    Return the id of the main trascript for a given gene.
    The data is from http://grch37.ensembl.org/
    '''
    data = EnsemblRelease(75)
    gene = data.genes_by_name(gene_name)
    gene_id = str(gene[0]).split(',')[0].split('=')[-1]
    gene_location = str(gene[0]).split('=')[-1].strip(')')
    url = 'http://grch37.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g={};r={}'.format(gene_id,gene_location)
    for line in urlopen(url):
        if '<tbody><tr><td class="bold">' in line:
            return line.split('">')[2].split('</a>')[0]

        
def get_exons_coord_by_gene_name(gene_name):
    '''str => OrderedDict({'exon_id':[coordinates]})
    Return an OrderedDict having as k the exon_id
    and as value a tuple containing the genomic coordinates ('chr',start,stop).        
    '''
    gene = data.genes_by_name(gene_name)
    gene_id = str(gene[0]).split(',')[0].split('=')[-1]
    gene_location = str(gene[0]).split('=')[-1].strip(')')
    gene_transcript = get_first_transcript_by_gene_name(gene_name).split('.')[0]
    table = OrderedDict()
    for exon_id in data.exon_ids_of_gene_id(gene_id):
        exon = data.exon_by_id(exon_id)
        coordinates = (exon.contig, exon.start, exon.end)
        table.update({exon_id:coordinates})
    return table


def get_exons_coord_by_gene_name(gene_name):
    '''string => OrderedDict
    .. code-block:: python
        >>> table = get_exons_coord_by_gene_name('TP53')
        >>> for k,v in table.items():
        ...    print(k,v)
            ENSE00002419584 ['7,579,721', '7,579,700']
    '''
    data = EnsemblRelease(75)
    gene = data.genes_by_name(gene_name)
    gene_id = str(gene[0]).split(',')[0].split('=')[-1]
    gene_location = str(gene[0]).split('=')[-1].strip(')')
    gene_transcript = get_first_transcript_by_gene_name(gene_name).split('.')[0]
    url = 'http://grch37.ensembl.org/Homo_sapiens/Transcript/Exons?db=core;g={};r={};t={}'.format(gene_id,gene_location,gene_transcript)
    str_html = get_html(url)
    html = ''
    for line in str_html.split('\n'):
        try:
            #print line
            html += str(line)+'\n'
        except UnicodeEncodeError:
            pass
    blocks = html.split('\n')
    table = OrderedDict()
    for exon_id in data.exon_ids_of_gene_id(gene_id):
        for i,txt in enumerate(blocks):
            if exon_id in txt:
                if exon_id not in table:
                    table.update({exon_id:[]})
                for item in txt.split('<td style="width:10%;text-align:left">')[1:-1]:
                    table[exon_id].append(item.split('</td>')[0])
    return table


def split_overlap(seq, size, overlap, is_dataframe=False):
    '''(seq,int,int) => [[...],[...],...]
    Split a sequence into chunks of a specific size and overlap.
    Works also on strings!
    It is very efficient for short sequences (len(seq()) <= 100).
    Set "is_dataframe=True" to split a pandas.DataFrame 
    
    Examples:
        >>> split_overlap(seq=list(range(10)),size=3,overlap=2)
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
        >>> split_overlap(seq=range(10),size=3,overlap=2)
        [range(0, 3), range(1, 4), range(2, 5), range(3, 6), range(4, 7), range(5, 8), range(6, 9), range(7, 10)]
    '''
    if size < 1 or overlap < 0:
        raise ValueError('"size must be >= 1 and overlap >= 0')
    
    result = []
    
    if is_dataframe:
        while True:
            if len(seq) <= size:
                result.append(seq)
                return result
            else:
                result.append(seq.iloc[:size])
                seq = seq.iloc[size-overlap:]
    
    else:
        while True:
            if len(seq) <= size:
                result.append(seq)
                return result
            else:
                result.append(seq[:size])
                seq = seq[size-overlap:]


def split_overlap_long(seq, size, overlap, is_dataframe=False):
    '''(seq,int,int) => generator
    Split a sequence into chunks of a specific size and overlap.
    Return a generator. It is very efficient for long sequences (len(seq()) > 100).
    https://stackoverflow.com/questions/48381870/a-better-way-to-split-a-sequence-in-chunks-with-overlaps
    Set "is_dataframe=True" to split a pandas.DataFrame
    Examples:
        >>> split_overlap_long(seq=list(range(10)),size=3,overlap=2)
        <generator object split_overlap_long at 0x10bc49d58>
        >>> list(split_overlap_long(seq=list(range(10)),size=3,overlap=2))
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
        >>> list(split_overlap_long(seq=range(10),size=3,overlap=2))
        [range(0, 3), range(1, 4), range(2, 5), range(3, 6), range(4, 7), range(5, 8), range(6, 9), range(7, 10)]
    '''       
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    if is_dataframe:
        for i in range(0, len(seq) - overlap, size - overlap):            
            yield seq.iloc[i:i + size]
    else:    
        for i in range(0, len(seq) - overlap, size - overlap):            
            yield seq[i:i + size]


def itr_split_overlap(iterable, size, overlap):
    '''(iterable,int,int) => generator
    Similar to long_split_overlap() but it works on any iterable.
    In case of long sequences, long_split_overlap() is more efficient
    but this function can handle potentially infinite iterables using deque().
    https://stackoverflow.com/questions/48381870/a-better-way-to-split-a-sequence-in-chunks-with-overlaps
    Warning: for range() and symilar, it behaves differently than split_overlap() and split_overlap_long()
    Examples:
        >>> list(itr_split_overlap(iterable=range(10),size=3,overlap=2))
        [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9)]
    '''
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    itr = iter(iterable)
    buf = deque(islice(itr, size), maxlen=size)

    chunk = None
    for chunk in iter(lambda: tuple(islice(itr, size - overlap)), ()):
        yield tuple(buf)
        buf.extend(chunk)

    rest = tuple(buf)

    if chunk:
        rest = rest[size - overlap - len(chunk):]

    yield rest
    

def split_overlap_df(df, size, overlap):
    '''(df,int,int) => generator
    Split a pandas.DataFrame into chunks of a specific size and overlap.
    '''       
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    for i in range(0, len(df) - overlap, size - overlap):            
        yield df.iloc[i:i + size]


def reorder_dict(d, keys):
    '''(dict,list) => OrderedDict
    Change the order of a dictionary's keys
    without copying the dictionary (save RAM!).
    Return an OrderedDict.
    '''
    tmp = OrderedDict()
    for k in keys:
        tmp[k] = d[k]
        del d[k] #this saves RAM
    return tmp

#test = OrderedDict({'1':1,'2':2,'4':4,'3':3})
#print(test)
#test2 = reorder_dict(test,['1','2','3','4'])
#print(test)
#print(test2)
#>>> OrderedDict([('2', 2), ('3', 3), ('4', 4), ('1', 1)])
#>>> OrderedDict()
#>>> OrderedDict([('1', 1), ('2', 2), ('3', 3), ('4', 4)])


def in_between(one_number, two_numbers):
    '''(int,list) => bool
    Return true if a number is in between two other numbers.
    Return False otherwise.
    '''
    if two_numbers[0] < two_numbers[1]:
        pass
    else:
        two_numbers = sorted(two_numbers)
    return two_numbers[0] <= one_number <= two_numbers[1]


def is_overlapping(svA, svB, limit=0.9):
    '''(list,list,float) => bool
    Check if two SV ovelaps for at least 90% (limit=0.9).
    svX = [chr1,brk1,chr2,brk2]
    '''
    
    # Step 1.
    # Select the breaks in order to have lower coordinates first
    if int(svA[1]) <= int(svA[3]):
        chr1_A = svA[0]
        brk1_A = int(svA[1])
        chr2_A = svA[2]
        brk2_A = int(svA[3])
    else:
        chr2_A = svA[0]
        brk2_A = svA[1]
        chr1_A = svA[2]
        brk1_A = svA[3]        
    
    if int(svB[1]) <= int(svB[3]):
        chr1_B = svB[0]
        brk1_B = int(svB[1])
        chr2_B = svB[2]
        brk2_B = int(svB[3])
    else:
        chr2_B = svB[0]
        brk2_B = int(svB[1])
        chr1_B = svB[2]
        brk1_B = int(svB[3])
        
    # Step 2.
    # Determine who is the longest
    # Return False immediately if the chromosomes are not the same.
    # This computation is reasonable only for sv on the same chormosome.
    if chr1_A == chr2_A and chr1_B == chr2_B and chr1_A == chr1_B:       
        len_A = brk2_A - brk1_A        
        len_B = brk2_B - brk1_B
        
        if len_A >= len_B:
            len_reference = len_A
            len_sample = len_B
        else:
            len_reference = len_B
            len_sample = len_A
        
        limit = round(len_reference * limit) # this is the minimum overlap the two sv need to share
                                     # to be considered overlapping
            
        # if the sample is smaller then the limit then there is no need to go further.
        # the sample segment will never share enough similarity with the reference.
        if len_sample < limit:
            return False
    else:
        return False
    
    # Step 3.
    # Determine if there is an overlap
    # >> There is an overlap if a least one of the break of an sv is in beetween the two breals of the other sv.
    overlapping = False
    for b in [brk1_A,brk2_A]:
        if in_between(b,[brk1_B,brk2_B]):
            overlapping = True
    for b in [brk1_B,brk2_B]:
        if in_between(b,[brk1_A,brk2_A]):
            overlapping = True
            
    if not overlapping:
        return False
    
    # Step 4.
    # Determine the lenght of the ovelapping part
    
    # easy case: if the points are all different then, if I sort the points,
    # the overlap is the region between points[1] and points[2]
    
    # |-----------------|             |---------------------|
    #         |--------------|              |-------------|
    points = sorted([brk1_A,brk2_A,brk1_B,brk2_B])
    if len(set(points)) == 4: # the points are all different
        overlap = points[2]-points[1]
        
    elif len(set(points)) == 3: #one point is in common
        # |-----------------|
        # |--------------|
        if points[0] == points[1]:
            overlap = points[3]-points[2]
            
        # |---------------------|
        #         |-------------|
        if points[2] == points[3]:
            overlap = points[2]-points[1]
            
        # |-----------------|
        #                   |-------------|
        if points[1] == points[2]:
            return False # there is no overlap
    else:
        # |-----------------|
        # |-----------------|
        return True # if two points are in common, then it is the very same sv
    
    if overlap >= limit:
        return True
    else:
        return False


def load_obj(file):
    '''
    Load a pickled object.
    Be aware that pickle is version dependent,
    i.e. objects dumped in Py3 cannot be loaded with Py2.
    '''
    
    try:
        with open(file,'rb') as f:
            obj = pickle.load(f)
        return obj
    except:
        return False


def save_obj(obj, file):
    '''
    Dump an object with pickle.
    Be aware that pickle is version dependent,
    i.e. objects dumped in Py3 cannot be loaded with Py2.
    '''
    try:
        with open(file,'wb') as f:
            pickle.dump(obj, f)
        print('Object saved to {}'.format(file))
        return True
    except:
        print('Error: Object not saved...')
        return False
    
#save_obj(hotspots_review,'hotspots_review_CIS.txt')


def query_encode(chromosome, start, end):
    '''
    Queries ENCODE via http://promoter.bx.psu.edu/ENCODE/search_human.php
    Parses the output and returns a dictionary of CIS elements found and the relative location.
    '''
    
    ## Regex setup
    re1='(chr{})'.format(chromosome) # The specific chromosome
    re2='(:)'    # Any Single Character ':'
    re3='(\\d+)' # Integer
    re4='(-)'    # Any Single Character '-'
    re5='(\\d+)' # Integer
    rg = re.compile(re1+re2+re3+re4+re5,re.IGNORECASE|re.DOTALL)

    ## Query ENCODE
    std_link = 'http://promoter.bx.psu.edu/ENCODE/get_human_cis_region.php?assembly=hg19&'
    query = std_link + 'chr=chr{}&start={}&end={}'.format(chromosome,start,end)
    print(query)
    html_doc = urlopen(query)
    html_txt = BeautifulSoup(html_doc, 'html.parser').get_text()
    data = html_txt.split('\n')

    ## Parse the output
    parsed = {}
    coordinates = [i for i, item_ in enumerate(data) if item_.strip() == 'Coordinate']
    elements = [data[i-2].split('  ')[-1].replace(': ','') for i in coordinates]
    blocks = [item for item in data if item[:3] == 'chr']
    print(elements)
    
    try:
        i = 0
        for item in elements:
            #print(i)
            try:
                txt = blocks[i]
                #print(txt)
                m = rg.findall(txt)
                bins = [''.join(item) for item in m]
                parsed.update({item:bins})
                i += 1
                print('found {}'.format(item))
            except:
                print('the field {} was empty'.format(item))
        return parsed
    except Exception as e:
        print('ENCODE query falied on chr{}:{}-{}'.format(chromosome, start, end))
        print(e)
        return False


def compare_patterns(pattA, pattB):
    '''(np.array, np.array) => float
    Compare two arrays point by point. 
    Return a "raw similarity score". 
    You may want to center the two patterns before compare them.
    >>> a  = np.array([1,2,3,4,5,6,5,4,3,2,1])
    >>> a1 = np.array([n+0.1 for n in a])
    >>> a2 = np.array([n+1 for n in a])
    >>> a3 = np.array([n+10 for n in a])
    >>> compare_patterns(a,a)
    99.999999999
    >>> compare_patterns(a,a1)
    95.69696969696969
    >>> compare_patterns(a,a2)
    56.96969696969697
    >>> compare_patterns(a2,a)
    72.33766233766234
    >>> compare_patterns(center(a),center(a2))
    99.999999999999943
    >>> compare_patterns(a,a3)
    -330.3030303030303
    '''

    if len(pattA) == len(pattB):
        deltas = []
        for i,pA in enumerate(pattA):
            deltas.append(100 - abs(delta_percent(pA,pattB[i])))

        similarity = sum(deltas)/len(pattA)
        return similarity
    else:
        raise ValueError('"pattA" and "pattB" must have same length.')

def compare_bins(dict_A,dict_B):
    '''(dict,dict) => dict, dict, dict
    Compares two dicts of bins.
    Returns the shared elements, the unique elements of A and the unique elements of B.
    The dicts shape is supposed to be like this:
        OrderedDict([('1',
                      ['23280000-23290000',
                       '24390000-24400000',
                       ...]),
                     ('2',
                      ['15970000-15980000',
                       '16020000-16030000',
                       ...]),
                     ('3',
                      ['610000-620000',
                       '3250000-3260000',
                       '6850000-6860000',
                       ...])}
    '''
    chrms = [str(x) for x in range(1,23)] + ['X','Y']
    
    shared = OrderedDict()
    unique_A = OrderedDict()
    unique_B = OrderedDict()
    for k in chrms:
        shared.update({k:[]})
        unique_A.update({k:[]})
        unique_B.update({k:[]})

        if k in dict_A and k in dict_B:
            for bin_ in dict_A[k]:
                if bin_ in dict_B[k]:
                    shared[k].append(bin_)
                else:
                    unique_A[k].append(bin_)
            for bin_ in dict_B[k]:
                if bin_ not in shared[k]:
                    unique_B[k].append(bin_)
        elif k not in dict_A:
            unique_B[k] = [bin_ for bin_ in dict_B[k]]
        
        elif k not in dict_B:
            unique_A[k] = [bin_ for bin_ in dict_A[k]]
            
    return shared, unique_A, unique_B


#To manage heavy files
def yield_file(infile):
    with open(infile, 'r') as f:
        for line in f:
            if line[0] not in ['#','\n',' ','']:
                yield line.strip()


#Downaload sequence from ensembl
def sequence_from_coordinates(chromosome, strand, start, end, ref_genome=37):
    '''
    Download the nucleotide sequence from the gene_name.
    '''
    Entrez.email = "a.marcozzi@umcutrecht.nl" # Always tell NCBI who you are
    
    if int(ref_genome) == 37:
        #GRCh37 from http://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.25/#/def_asm_Primary_Assembly
        NCBI_IDS = {'1':'NC_000001.10','2':'NC_000002.11','3':'NC_000003.11','4':'NC_000004.11',
                    '5':'NC_000005.9','6':'NC_000006.11','7':'NC_000007.13','8':'NC_000008.10',
                    '9':'NC_000009.11','10':'NC_000010.10','11':'NC_000011.9','12':'NC_000012.11',
                    '13':'NC_000013.10','14':'NC_000014.8','15':'NC_000015.9','16':'NC_000016.9',
                    '17':'NC_000017.10','18':'NC_000018.9','19':'NC_000019.9','20':'NC_000020.10',
                    '21':'NC_000021.8','22':'NC_000022.10','X':'NC_000023.10','Y':'NC_000024.9'}
    elif int(ref_genome) == 38:
        #GRCh38 from https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.38
        NCBI_IDS = {'1':'NC_000001.11','2':'NC_000002.12','3':'NC_000003.12','4':'NC_000004.12',
                    '5':'NC_000005.10','6':'NC_000006.12','7':'NC_000007.14','8':'NC_000008.11',
                    '9':'NC_000009.12','10':'NC_000010.11','11':'NC_000011.10','12':'NC_000012.12',
                    '13':'NC_000013.11','14':'NC_000014.9','15':'NC_000015.10','16':'NC_000016.10',
                    '17':'NC_000017.11','18':'NC_000018.10','19':'NC_000019.10','20':'NC_000020.11',
                    '21':'NC_000021.9','22':'NC_000022.11','X':'NC_000023.11','Y':'NC_000024.10'}
        
  
    try:        
        handle = Entrez.efetch(db="nucleotide", 
                               id=NCBI_IDS[str(chromosome)], 
                               rettype="fasta", 
                               strand=strand, #"1" for the plus strand and "2" for the minus strand.
                               seq_start=start,
                               seq_stop=end)
        record = SeqIO.read(handle, "fasta")
        handle.close()
        sequence = str(record.seq)
        return sequence
    except ValueError:
        print('ValueError: no sequence found in NCBI')
        return False


#GC content calculator    
def gc_content(sequence, percent=True):
    '''
    Return the GC content of a sequence.
    '''
    sequence = sequence.upper()
    g = sequence.count("G")
    c = sequence.count("C")
    t = sequence.count("T")
    a = sequence.count("A")
    gc_count = g+c
    total_bases_count = g+c+t+a
    if total_bases_count == 0:
        print('Error in gc_content(sequence): sequence may contain only Ns')
        return None
    
    try:
        gc_fraction = float(gc_count) / total_bases_count
    except Exception as e:
        print(e)
        print(sequence)
    
    if percent:
        return gc_fraction * 100
    else:
        return gc_fraction
    
    
       
##Flexibility calculator##  
#requires stabflex3.py




#Endpoint function to calculate the flexibility of a given sequence
def dna_flex(sequence, window_size=500, step_zize=100, verbose=False):
    '''(str,int,int,bool) => list_of_tuples
    Calculate the flexibility index of a sequence.
    Return a list of tuples.
    Each tuple contains the bin's coordinates
    and the calculated flexibility of that bin.
    Example:
        dna_flex(seq_a,500,100)
    >>> [('0-500', 9.7),('100-600', 9.77),...]
    '''
    if verbose:
        print("Algorithm window size : %d" % window_size)
        print("Algorithm window step : %d" % step_zize)
        print("Sequence has {} bases".format(len(self.seq)))
        
    algorithm = myFlex(sequence,window_size,step_zize)
    flexibility_result = algorithm.analyse(flexibility_data)
    
    return flexibility_result.report(verbose)


##Repeats scanner##

#G-quadruplex
def g4_scanner(sequence):
    '''
    G-quadruplex motif scanner.
    Scan a sequence for the presence of the regex motif:
        [G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}
        Reference: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1636468/
    Return two callable iterators.
    The first one contains G4 found on the + strand.
    The second contains the complementary G4 found on the + strand, i.e. a G4 in the - strand.
    '''
    #forward G4
    pattern_f = '[G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}'
    result_f = re.finditer(pattern_f, sequence)
    #reverse G4
    pattern_r = '[C]{3,5}[ACGT]{1,7}[C]{3,5}[ACGT]{1,7}[C]{3,5}[ACGT]{1,7}[C]{3,5}'
    result_r = re.finditer(pattern_r, sequence)
    
    return result_f, result_r


#Repeat-masker
def parse_RepeatMasker(infile="RepeatMasker.txt", rep_type='class'):
    '''
    Parse RepeatMasker.txt and return a dict of bins for each chromosome
    and a set of repeats found on that bin.
    
    dict = {'chromosome':{'bin':set(repeats)}}
    '''
    
    chromosomes = [str(c) for c in range(1,23)]+['X','Y']
    result = {}
    
    if rep_type == 'name':
        idx = 10 #repName
    elif rep_type == 'class':
        idx = 11 #repClass
    elif rep_type == 'family':
        idx = 12 #repFamily
    else:
        raise NameError('Invalid rep_type "{}". Expected "class","family" or "name"'.format(rep_type))
            
    #RepeatMasker.txt is around 500MB!
    for line in yield_file(infile):
        data = line.split('\t')
        chromosome = data[5].replace('chr','')
        start = data[6]
        end = data[7]
        bin_ = '{}-{}'.format(start,end)
        repeat = data[idx].replace('?','')
        
        if chromosome in chromosomes:
            if chromosome not in result:
                result.update({chromosome:{bin_:set([repeat])}})
            else:
                if bin_ not in result[chromosome]:
                    result[chromosome].update({bin_:set([repeat])})
                else:
                    result[chromosome][bin_].add(repeat)
                
    return result


def next_day(d='2012-12-04'):
    '''Return the next day in the calendar.'''
    Y,M,D = d.split('-')
    t = datetime.date(int(Y),int(M),int(D))
    _next = t + datetime.timedelta(1)
    return str(_next)
# next_day('2012-12-31')
# >>> '2013-01-01'


def previous_day(d='2012-12-04'):
    '''Return the previous day in the calendar.'''
    Y,M,D = d.split('-')
    t = datetime.date(int(Y),int(M),int(D))
    _prev = t + datetime.timedelta(-1)
    return str(_prev)
# previous_day('2013-01-01')
# >>> '2012-12-31'


def intersect(list1, list2):
    '''(list,list) => list
    Return the intersection of two lists, i.e. the item in common.
    '''
    return [item for item in list2 if item in list1]


def annotate_fusion_genes(dataset_file):
    '''
    Uses FusionGenes_Annotation.pl to find fusion genes in the dataset.
    Generates a new file containing all the annotations.
    '''
    start = time.time()
    print('annotating', dataset_file, '...')
    raw_output = run_perl('FusionGenes_Annotation.pl', dataset_file)
    raw_list = str(raw_output)[2:].split('\\n')
    outfile = dataset_file[:-4] + '_annotated.txt'
    with open(outfile, 'w') as outfile:
        line_counter = 0
        header = ['##ID', 'ChrA', 'StartA', 'EndA', 'ChrB', 'StartB', 'EndB', 'CnvType', 'Orientation',
                  'GeneA', 'StrandA', 'LastExonA', 'TotalExonsA', 'PhaseA',
                  'GeneB', 'StrandB', 'LastExonB', 'TotalExonsB', 'PhaseB',
                  'InFrame', 'InPhase']
        outfile.write(list_to_line(header, '\t') + '\n')
        for item in raw_list:
            cleaned_item = item.split('\\t')
            if len(cleaned_item) > 10: # FusionGenes_Annotation.pl return the data twice. We kepp the annotated one.
                outfile.write(list_to_line(cleaned_item, '\t') + '\n')
                line_counter += 1
    print('succesfully annotated',line_counter, 'breakpoints from', dataset_file, 'in', time.time()-start, 'seconds') 
    # track threads
    try:
        global running_threads
        running_threads -= 1
    except:
        pass
# dataset_file = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/Decipher-DeletionsOnly.txt'
# annotate_fusion_genes(dataset_file)


def blastn(input_fasta_file, db_path='/Users/amarcozzi/Desktop/BLAST_DB/',db_name='human_genomic',out_file='blastn_out.xml'):
    '''
    Run blastn on the local machine using a local database.
    Requires NCBI BLAST+ to be installed. http://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download
    Takes a fasta file as input and writes the output in an XML file.
    '''
    db = db_path + db_name
    blastn_cline = NcbiblastnCommandline(query=input_fasta_file, db=db, evalue=0.001, outfmt=5, out=out_file)
    print(blastn_cline)
    stdout, stderr = blastn_cline()
# to be tested


def check_line(line, unexpected_char=['\n','',' ','#']):
    '''
    Check if the line starts with an unexpected character.
    If so, return False, else True
    '''
    for item in unexpected_char:
        if line.startswith(item):
            return False
    return True


def dice_coefficient(sequence_a, sequence_b):
    '''(str, str) => float
    Return the dice cofficient of two sequences.
    '''
    a = sequence_a
    b = sequence_b
    if not len(a) or not len(b): return 0.0
    # quick case for true duplicates
    if a == b: return 1.0
    # if a != b, and a or b are single chars, then they can't possibly match
    if len(a) == 1 or len(b) == 1: return 0.0
    
    # list comprehension, preferred over list.append() '''
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
    
    a_bigram_list.sort()
    b_bigram_list.sort()
    
    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
    
    score = float(matches)/float(lena + lenb)
    return score


def find_path(graph, start, end, path=[]):
    '''
    Find a path between two nodes in a graph.
    Works on graphs like this:
            graph ={'A': ['B', 'C'],
                    'B': ['C', 'D'],
                    'C': ['D'],
                    'D': ['C'],
                    'E': ['F'],
                    'F': ['C']}
    '''
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
        if newpath: return newpath
    return None


def find_all_paths(graph, start, end, path=[]):
    '''
    Find all paths between two nodes of a graph.
    Works on graphs like this:
            graph ={'A': ['B', 'C'],
                    'B': ['C', 'D'],
                    'C': ['D'],
                    'D': ['C'],
                    'E': ['F'],
                    'F': ['C']}
    '''
    path = path + [start]
    if start == end:
        return [path]
    if not graph.has_key(start):
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def find_shortest_path(graph, start, end, path=[]):
    '''
    Find the shortest path between two nodes of a graph.
    Works on graphs like this:
            graph ={'A': ['B', 'C'],
                    'B': ['C', 'D'],
                    'C': ['D'],
                    'D': ['C'],
                    'E': ['F'],
                    'F': ['C']}
    '''
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


# ##
# graph = {'A': ['B', 'C'],
#        'B': ['C', 'D'],
#        'C': ['D'],
#        'D': ['C'],
#        'E': ['F'],
#        'F': ['C']}

# >>> find_path(graph, 'A', 'D')
#     ['A', 'B', 'C', 'D']

# >>> find_all_paths(graph, 'A', 'D')
#     [['A', 'B', 'C', 'D'], ['A', 'B', 'D'], ['A', 'C', 'D']]

# >>> find_shortest_path(graph, 'A', 'D')
#     ['A', 'C', 'D']

def gen_rnd_string(length):
    '''
    Return a string of uppercase and lowercase ascii letters.
    '''
    
    s = [l for l in string.ascii_letters]
    random.shuffle(s)
    s = ''.join(s[:length])
    return s

def gene_synonyms(gene_name):
    '''str => list()
    Queries http://rest.genenames.org and returns a list of synonyms of gene_name.
    Returns None if no synonym was found.
    '''


    result = []
    headers = {'Accept': 'application/json'}

    uri = 'http://rest.genenames.org'
    path = '/search/{}'.format(gene_name)

    target = urlparse(uri+path)
    method = 'GET'
    body = ''

    h = http.Http()

    response, content = h.request(
                                    target.geturl(),
                                    method,
                                    body,
                                    headers )

    if response['status'] == '200':
        # assume that content is a json reply
        # parse content with the json module 
        data = json.loads(content.decode('utf8'))
        for item in data['response']['docs']:
            result.append(item['symbol'])
        return result
     
    else:
        print('Error detected: ' + response['status'])
        return None
#print(gene_synonyms('MLL3'))

def string_to_number(s):
    '''
    Convert a bytes string into a single number.
    Example:
        >>> string_to_number('foo bar baz')
        147948829660780569073512294
    '''
    return int.from_bytes(s.encode(), 'little')


def number_to_string(n):
    '''
    Convert a number into a bytes string.
    Example:
        >>> number_to_string(147948829660780569073512294)
        'foo bar baz'
    '''
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()
#x = 147948829660780569073512294
#number_to_string(x)
#>>> 'foo bar baz'
 
def determine_average_breaks_distance(dataset): # tested only for deletion/duplication
    '''
    Evaluate the average distance among breaks in a dataset.
    '''
    data = extract_data(dataset, columns=[1,2,4,5], verbose=False)
    to_average = []
    for item in data:
        if item[0] == item[2]:
            to_average.append(int(item[3])-int(item[1]))
    return sum(to_average)/len(to_average)
#print(determine_average_breaks_distance('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/random/sorted/rnd_dataset_100_annotated_sorted.txt'))


def dict_overview(dictionary, how_many_keys, indent=False):
    '''
    Prints out how_many_elements of the target dictionary.
    Useful to have a quick look at the structure of a dictionary.
    '''
    
    ks = list(islice(dictionary, how_many_keys))
    for k in ks:
        if indent:
            print(f'{k}\n\t{dictionary[k]}')
        else:
            print(f'{k}\t{dictionary[k]}')


def download_human_genome(build='GRCh37', entrez_usr_email="A.E.vanvlimmeren@students.uu.nl"): #beta: works properly only forGRCh37
    '''
    Download the Human genome from enterez.
    Save each chromosome in a separate txt file.
    '''
    

    Entrez.email = entrez_usr_email

    #Last available version
    NCBI_IDS = {'1':"NC_000001", '2':"NC_000002",'3':"NC_000003",'4':"NC_000004",
                '5':"NC_000005",'6':"NC_000006",'7':"NC_000007", '8':"NC_000008",
                '9':"NC_000009", '10':"NC_000010", '11':"NC_000011", '12':"NC_000012",
                '13':"NC_000013",'14':"NC_000014", '15':"NC_000015", '16':"NC_000016", 
                '17':"NC_000017", '18':"NC_000018", '19':"NC_000019", '20':"NC_000020",
                '21':"NC_000021", '22':"NC_000022", 'X':"NC_000023", 'Y':"NC_000024"}


    #GRCh37 from http://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.25/#/def_asm_Primary_Assembly
    NCBI_IDS_GRCh37 = { '1':'NC_000001.10','2':'NC_000002.11','3':'NC_000003.11','4':'NC_000004.11',
                        '5':'NC_000005.9','6':'NC_000006.11','7':'NC_000007.13','8':'NC_000008.10',
                        '9':'NC_000009.11','10':'NC_000010.10','11':'NC_000011.9','12':'NC_000012.11',
                        '13':'NC_000013.10','14':'NC_000014.8','15':'NC_000015.9','16':'NC_000016.9',
                        '17':'NC_000017.10','18':'NC_000018.9','19':'NC_000019.9','20':'NC_000020.10',
                        '21':'NC_000021.8','22':'NC_000022.10','X':'NC_000023.10','Y':'NC_000024.9'}
    

    CHR_LENGTHS_GRCh37 = {  '1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
                            '5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
                            '9' :141213431,'10':135534747,'11':135006516,'12':133851895,
                            '13':115169878,'14':107349540,'15':102531392,'16':90354753,
                            '17':81195210,'18':78077248,'19':59128983,'20':63025520,
                            '21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}


    if build == 'GRCh37':
        NCBI_IDS = NCBI_IDS_GRCh37
        CHR_LENGTHS = CHR_LENGTHS_GRCh37
    else:
        print('This function only work with genome build GRCh37 fow now...')
        return False


    
    for chromosome, nc_id in NCBI_IDS.items():
        print(f'downloading {nc_id}')
        length = CHR_LENGTHS[chromosome]
        sequence = False

        try:
                 # Always tell NCBI who you are
            handle = Entrez.efetch(db="nucleotide", 
                                   id=nc_id, 
                                   rettype="fasta", 
                                   strand=1, 
                                   seq_start=0, #this is to obtain actual start coordinates from the index
                                   seq_stop=length) # this is the end of the chromosome
            record = SeqIO.read(handle, "fasta")
            handle.close()
            sequence = str(record.seq)

        except ValueError:
            print('ValueError: no sequence found in NCBI')

        with open('sequence_{}.txt'.format(chromosome), 'w') as f:
            f.write(sequence)   


def exponential_range(start=0,end=10000,base=10):
    '''
    Generates a range of integer that grow exponentially.
    Example: list(exp_range(0,100000,2))
    Output :[0,
             2,
             4,
             8,
             16,
             32,
             64,
             128,
             256,
             512,
             1024,
             2048,
             4096,
             8192,
             16384,
             32768,
             65536]
    '''

    if end/base < base:
        raise ValueError('"end" must be at least "base**2"')
    result = []
     
    new_start = start
    new_end = base**2
    new_base = base
    
    while new_start < end:
        result.append(range(new_start,new_end,new_base))
        
        new_start = new_end
        new_end = new_start*base
        new_base = new_base*base
              
    #print(result)
    for item in result:    
        for i in item:
            yield i
##list(exp_range(0,100000,10))


def extract_data(infile, columns=[3,0,1,2,5], header='##', skip_lines_starting_with='#', data_separator='\t', verbose=False ):
    '''
    Extract data from a file. Returns a list of tuples. 
    Each tuple contains the data extracted from one line of the file
    in the indicated columns and with the indicated order.
    '''
    
    extracted_data = []
    header_list = []
    header_flag = 0
    line_counter = 0

    with open(infile) as infile:
        lines = infile.readlines()

    for line in lines: # yield_file(infile) can be used instead
        line_counter += 1

        if line[:len(header)] == header: # checks the header
            header_list = line_to_list(line[len(header):], data_separator)
            header_flag += 1
            if header_flag > 1:
                raise ValueError('More than one line seems to contain the header identificator "' + header + '".')
        elif line[0] == skip_lines_starting_with or line == '' or line == '\n': # skips comments and blank lines
            pass
        else:
            list_ = line_to_list(line, data_separator)
            reduced_list=[]
            for item in columns:
                reduced_list.append(list_[item])
            extracted_data.append(tuple(reduced_list))

    if verbose == True: # Prints out a brief summary
        print('Data extracted from', infile)
        print('Header =', header_list)
        print('Total lines =', line_counter)

    return extracted_data
# extract_data('tables/clinvarCnv.txt', columns=[3,0,1,2,5], header='##', skip_lines_starting_with='#', data_separator='\t', verbose=True)


def extract_Toronto(infile, outfile):
    '''
    Ad hoc function to extract deletions and duplications out of the Toronto Genetic Variants Database.
    Returns a file ready to be annotated with FusionGenes_Annotation.pl .
    '''
    # Extract data from infile
    # Columns are: ID, Chr, Start, End, CNV_Type
    raw_data = extract_data(infile, columns=[0,1,2,3,5], verbose=True )

    # Take only deletions and duplications
    filtered_data = []
    for data in raw_data:
        if "deletion" in data or 'duplication' in data:
            filtered_data.append(data)
    print('len(row_data)      :',len(raw_data))
    print('len(filtered_data) :',len(filtered_data))


    # Write filtered_data to a text file
    header = ['##ID','ChrA','StartA','EndA','ChrB','StartB','EndB','CnvType','Orientation']
    with open(outfile, 'w') as outfile:
        outfile.write(list_to_line(header, '\t') + '\n')
        for item in filtered_data:
            if item[-1] == 'duplication':
                orientation = 'HT'
            elif item[-1] == 'deletion':
                orientation = 'TH'
            else:
                print('ERROR: unable to determine "Orientation"...')
            list_ = [item[0],item[1],item[2],item[2],item[1],item[3],item[3],item[-1].upper(),orientation]
            outfile.write(list_to_line(list_, '\t') + '\n')
    print('Done')
# infile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/GRCh37_hg19_variants_2014-10-16.txt'
# outfile = infile[:-4]+'_DelDupOnly.txt'
# extract_Toronto(infile, outfile)


def extract_Decipher(infile, outfile):
    '''
    Ad hoc function to extract deletions and duplications out of the Decipher Database.
    Returns a file ready to be annotated with FusionGenes_Annotation.pl .
    '''
    # Extract data from infile
    # Columns are: ID, Chr, Start, End, CNV_Type(here expressed as "mean_ratio")
    raw_data = extract_data(infile, columns=[0,3,1,2,4], verbose=True )
    header = ['##ID','ChrA','StartA','EndA','ChrB','StartB','EndB','CnvType','Orientation']
    with open(outfile, 'w') as outfile:
        outfile.write(list_to_line(header, '\t') + '\n')
        for item in raw_data:
            # Convert mean_ratio to CnvType
            if float(item[-1]) > 0:
                CnvType = 'DUPLICATION'
                orientation = 'HT'
            elif float(item[-1]) < 0:
                CnvType = 'DELETION'
                orientation = 'TH'
            else:
                print('ERROR: unable to determine "Orientation"...')
            # Write output
            list_ = [item[0],item[1],item[2],item[2],item[1],item[3],item[3],CnvType,orientation]
            outfile.write(list_to_line(list_, '\t') + '\n')
    print('Done')
# infile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/decipher-hg19_15-01-30.txt'
# outfile = infile[:-4]+'_DelDupOnly.txt'
# extract_Decipher(infile, outfile)


def extract_dgvMerged(infile, outfile):
    '''
    Ad hoc function to extract deletions and losses out of the dgvMerged database.
    Returns a file ready to be annotated with FusionGenes_Annotation.pl .
    '''
    #original_header = '##bin   chrom   chromStart  chromEnd    name    score   strand  thickStart  thickEnd    itemRgb varType reference   pubMedId    method  platform    mergedVariants  supportingVariants  sampleSize  observedGains   observedLosses  cohortDescription   genes   samples'
                    #   [0]     [1]     [2]         [3]         [4]     [5]     [6]     [7]         [8]         [9]     [10]    [11]        [12]        [13]    [14]        [15]            [16]                [17]        [18]            [19]            [20]                [21]    [22]
    raw_data = extract_data(infile, columns=[4,1,2,3,10], header='##', skip_lines_starting_with='#', data_separator='\t', verbose=False )

    # Take only deletions and losses
    filtered_data = []
    for data in raw_data:
        if "Deletion" in data or 'Loss' in data:
            filtered_data.append(data)
    print('len(row_data)      :',len(raw_data))
    print('len(filtered_data) :',len(filtered_data))

    # Write filtered_data to a text file
    header = ['##ID','ChrA','StartA','EndA','ChrB','StartB','EndB','CnvType','Orientation']
    with open(outfile, 'w') as outfile:
        outfile.write(list_to_line(header, '\t') + '\n')
        for item in filtered_data:
            if item[-1] == 'Deletion' or item[-1] == 'Loss':
                cnv_type = 'DELETION'
                orientation = 'HT'
            # elif item[-1] == 'deletion':
            #   orientation = 'TH'
            else:
                print('ERROR: unable to determine "Orientation"...')
            list_ = [item[0],item[1][3:],item[2],item[2],item[1][3:],item[3],item[3],cnv_type,orientation]
            outfile.write(list_to_line(list_, '\t') + '\n')
    print('Done')
# ## Extract deletions and Losses from dgvMerged
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks'
# file_name = 'dgvMerged.txt'
# infile = folder + '/' + file_name
# outfile = folder + '/' + 'dgvMerged-DeletionsOnly.txt'
# extract_dgvMerged(infile, outfile)
# ## annotate
# dataset_file = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/dgvMerged-DeletionsOnly.txt'
# annotate_fusion_genes(dataset_file)


def fill_and_sort(pandas_chrSeries):
    '''incomplete pandas.Series => complete and sorted pandas.Series
    Given a pandas.Series in which the first argument is the chromosome name
    and the second argument is a count " [('1', 61), ('3', 28), ..., ('X', 29)]"
    This function returns a new (sorted by chromosome) series with the missing chromosome included as ('Chr_name',0).
    
    This is useful when creating series out of subsets grouped by Chr.
    If the Chr does not contains any event, then it will be excluded from the subset.
    However, expecially for plotting reasons, you may want to have ('Chr',0) in you list instead of a missing Chr.
    
    Example.
    > series = [('1', 61), ('3', 28), ..., ('X', 29)] # in this Series Chr_2 and Chr_Y are missing.
    > fill_and_sort(series)
    >>> [('1', 61), ('2',0), ('3', 28), ..., ('X', 29), ('Y',0)] # this Series have all the chromosomes
    '''
    
    # add missing ChrA
    CHROMOSOMES = [str(c) for c in range(1,23)]+['X','Y']
    chr_list = CHROMOSOMES[:]
    complete_series = []
    for item in pandas_chrSeries.iteritems():
        chr_list.remove(item[0])
        complete_series.append(item)
    for item in chr_list:
        complete_series.append((item,0))
    
    # sort by chromosome
    sorted_ = []
    for item in CHROMOSOMES:
        for _item in complete_series:
            if _item[0]==item:
                sorted_.append(_item[1])
    return pd.Series(sorted_, index=CHROMOSOMES)
# counts = [50,9,45,6]
# pandas_chrSeries = pd.Series(counts, index=['1','4','X','10'])
# print(pandas_chrSeries)
# good_series = fill_and_sort(pandas_chrSeries)
# print(good_series)


def find(string, char):
    '''
    Looks for a character in a sctring and returns its index.
    '''
    # Compared to string.find(), it returns ALL the indexes, not only the first one.
    return [index for index, letter in enumerate(string) if letter == char]
# print(find('alessio', 's'))

def filter_out(word, infile, outfile):
    '''
    Reads a file line by line
    and writes an output file containing only
    the lines that DO NOT contains 'word'.
    '''
    print('Filtering out lines containing',word,'...')
    with open(infile, 'r') as infile:
        lines = infile.readlines()
    with open(outfile, 'w') as outfile:
        for line in lines: # yield_file(infile) can be used instead
            if word not in line:
                outfile.write(line)
    print('Done')
# infile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/Decipher_DelDupOnly.txt'
# outfile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/Decipher-DeletionsOnly.txt'
# filter_out('DUPLICATION',infile, outfile)

def flatten2(l):
    '''
    Flat an irregular iterable to a list.
    Python >= 2.6 version.
    '''
    for item in l:
        if isinstance(item, collections.Iterable) and not isinstance(item, basestring):
            for sub in flatten(item):
                yield sub
        else:
            yield item


def flatten(l):
    '''
    Flat an irregular iterable to a list.
    Python >= 3.3 version.
    '''
    for item in l:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

def gene_synonyms(gene_name):
    '''str => list()
    Queries http://rest.genenames.org and http://www.ncbi.nlm.nih.gov/ to figure out the best synonym of gene_name.
    '''
    
    

    result = []
    tmp = []
    headers = {'Accept': 'application/json'}

    uri = 'http://rest.genenames.org'
    path = '/search/{}'.format(gene_name)
    html_doc = urlopen('http://www.ncbi.nlm.nih.gov/gene/?term={}[sym]'.format(gene_name))
    html_txt = BeautifulSoup(html_doc, 'html.parser').get_text()


    target = urlparse(uri+path)
    method = 'GET'
    body = ''

    h = http.Http()

    response, content = h.request(
                                    target.geturl(),
                                    method,
                                    body,
                                    headers )

    if response['status'] == '200':
        # assume that content is a json reply
        # parse content with the json module 
        data = json.loads(content.decode('utf8'))
        for item in data['response']['docs']:
            tmp.append(item['symbol'])
     
    else:
        print('Error detected: ' + response['status'])
        return None

    if len(tmp) > 1:
        for gene in tmp:
            if gene in html_txt:
                result.append(gene)
        return result
    else:
        return tmp
#print(gene_synonyms('MLL3'))

def gen_controls(how_many, chromosome, GapTable_file,outfile):
    global running_threads # in case of multithreading
    list_brkps = gen_rnd_single_break(how_many, chromosome, GapTable_file, verbose=False)
    with open(outfile,'w') as f:
        for item in list_brkps:
            f.write(list_to_line(item,'\t')+'\n')
    running_threads -= 1 # in case of multithreading
# # Generate controls
# import time
# from threading import Thread
# threads = 0
# running_threads = 0
# max_simultaneous_threads = 20
# how_many=9045
# chromosome='9'
# GapTable_file='/Users/alec/Desktop/UMCU_Backup/Projects/Anne_Project/current_brkps_DB/out_ALL_gap.txt'
# while threads < 100:
#   while running_threads >= max_simultaneous_threads:
#       time.sleep(1)
#   running_threads += 1
#   outfile = '/Users/alec/Desktop/UMCU_Backup/Projects/Anne_Project/current_brkps_DB/out_chr9_control_'+str(threads)+'.txt'
#   print('thread', threads, '|', 'running threads:',running_threads)
#   Thread(target=gen_controls, args=(how_many,chromosome,GapTable_file,outfile)).start()
#   threads += 1

def gen_control_dataset(real_dataset, suffix='_control.txt'):# tested only for deletion/duplication
    '''
    Generates a control dataset ad hoc.
    Takes as input an existing dataset and generates breaks
    in the same chromosomes and with the same distance (+-1bp),
    the position are however randomized.
    '''
    real_data = extract_data(real_dataset, columns=[1,2,4,5,7,8], verbose=False)
    control_data = []
    _id_list = []
    for item in real_data:
        if item[0] == item[2]: # ChrA == ChrB
            
            # generate a unique id
            _id = gen_rnd_id(16)
            while _id in _id_list:
                    _id = gen_rnd_id(16)
            _id_list.append(_id)

            chromosome = item[0]
            distance = int(item[3])-int(item[1]) # 
            cnv_type = item[4]
            orientation = item[5]
            breaks = gen_rnd_breaks(how_many=1, chromosome=chromosome,
                                    min_distance=distance-1, max_distance=distance+1,
                                    GapTable_file='tables/gap.txt')
            print(breaks)
            control_data.append([_id,chromosome,breaks[0][1],breaks[0][1],chromosome,breaks[0][2],
                                 breaks[0][2],cnv_type,orientation])
        else:
            print(item[0],'is no equal to',item[2],'I am skipping these breaks')
    
    header = ['##ID', 'ChrA', 'StartA', 'EndA', 'ChrB', 'StartB', 'EndB', 'CnvType', 'Orientation']
    
    filename = real_dataset[:-4]+ suffix
    with open(filename,'w') as outfile:
        outfile.write(list_to_line(header, '\t') + '\n')
        for item in control_data:
            line = list_to_line(item, '\t')
            print(line)
            outfile.write(line + '\n')

    print('Data written in',filename)
# gen_control_dataset('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/raw/clinvarCnv-DeletionsOnly.txt')

def gen_gap_table(infile='/Users/amarcozzi/Desktop/All_breakpoints_HG19_final.txt', outfile='/Users/amarcozzi/Desktop/All_breakpoints_HG19_gap.txt', resolution=10000):
    '''
    Generates a file containing a list of coordinates 
    for wich no brakpoints have been found in the input file.
    '''
    # Global constants
    CHROMOSOMES = [str(c) for c in range(1,23)]+['X','Y']
    # length of chromosomes based on GRCh37 (Data source: Ensembl genome browser release 68, July 2012)
    # http://jul2012.archive.ensembl.org/Homo_sapiens/Location/Chromosome?r=1:1-1000000
    # http://grch37.ensembl.org/Homo_sapiens/Location/Chromosome?r=1:24626643-24726643
    CHR_LENGTHS = {'1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
                   '5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
                   '9' :141213431,'10':135534747,'11':135006516,'12':133851895,
                   '13':115169878,'14':107349540,'15':102531392,'16':90354753,
                   '17':81195210,'18':78077248,'19':59128983,'20':63025520,
                   '21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}
    gap_list = []
    for Chr in CHROMOSOMES:
        print('-----------------------------------------------------')
        print('Analyzing breakpoints in chromosome',Chr)
        length = CHR_LENGTHS[Chr]
        # determine the intervals given the chromosome length and the resolution
        x_ax = [] # data holder
        y_ax = [] # stores breakpoint counts per inteval
        breakpoint_list = []
        
        # # Extract data from infile, chromosome by chromosome
        # with open(infile, 'r') as f:
        #   lines = f.readlines()
        #   for line in lines: # yield_file(infile) can be used instead
        #       if line.startswith('chr'+Chr+':'):
        #           tmp = line.split(':')
        #           breakpoint = tmp[1].split('-')[0]
        #           breakpoint_list.append(int(breakpoint))
        # print(len(breakpoint_list),'breakpoints found...')

        with open(infile, 'r') as f:
            #lines = f.readlines()
            for line in f:#lines: # yield_file(infile) can be used instead
                if line.startswith(Chr+'\t'):
                    tmp = line_to_list(line,'\t')
                    breakpoint = tmp[1]
                    breakpoint_list.append(int(breakpoint))
        print(len(breakpoint_list),'breakpoints found...')

        for item in range(resolution,length+resolution,resolution):
            x_ax.append(item)
        print('Interval list:',len(x_ax), 'at',resolution,'bases resolution')

        for interval in x_ax:
            count = 0
            to_remove = []
            for breakpoint in breakpoint_list:
                if breakpoint <= interval:
                    count += 1
                    to_remove.append(breakpoint)
            y_ax.append(count)

            for item in to_remove:
                try:
                    breakpoint_list.remove(item)
                except:
                    print('Error',item)

        counter = 0
        for idx,count_ in enumerate(y_ax):
            if count_ == 0:
                gap = x_ax[idx]
                gap_list.append((Chr,gap))
                counter += 1
        print('Found', counter,'gaps in chromosome',Chr,'\n')

    with open(outfile, 'w') as f:
        f.write('#Gap table at '+str(resolution)+' bases resolution based on '+infile+'\n')
        f.write('##chrom'+'\t'+'chromStart'+'\t'+'chromEnd'+'\n')

        for item in gap_list:
            line = 'chr'+str(item[0])+'\t'+str(item[1]-resolution)+'\t'+str(item[1])
            f.write(line+'\n')
# import time
# start = time.time()
# gen_gap_table()
# print('Done in',time.time()-start,'seconds')
## Generate a gap table file
# import time
# start = time.time()
# gen_gap_table(infile='/Users/amarcozzi/Desktop/current_brkps_DB/out_ALL.txt', outfile='/Users/amarcozzi/Desktop/current_brkps_DB/out_ALL_gap.txt', resolution=10000)
# print('Done in',time.time()-start,'seconds')

def gen_multiple_controls(real_dataset, how_many):
    '''
    Generates how_many control datasets.
    '''
    n=0
    while n < how_many:
        suffix = '_control_'+str(n)+'.txt'
        #real_dataset = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/raw/dataset_1b.txt'
        gen_control_dataset(real_dataset,suffix)
        n+=1
    print(n,'datasets have been generated')
# gen_multiple_controls('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/raw/dataset_4.txt',1000)
# ## Generate multiple controls of datasets found in a folder
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/random'
# for item in list_of_files(folder,'txt'):
#   gen_multiple_controls(item,1000)

def gen_deletion_dataset_from_breaks(list_of_breaks, outfile, ID_already=False):
    '''Genrates a proper deletion dataset file out of a list of breaks '''
    # Var names are not pythonic but I think it is better for readibility
    header = ['##ID', 'ChrA', 'StartA', 'EndA', 'ChrB', 'StartB', 'EndB', 'CnvType', 'Orientation']
    ID_list = [] # to check if the ID is already present
    print('writing breakpoints to', outfile, '..........')
    with open(outfile, 'w') as outfile:
        outfile.write(list_to_line(header, '\t') + '\n')
        for item in list_of_breaks:
            if ID_already == False: # the braks do not have an ID
                while True: # checks ID
                    ID = gen_rnd_id(8)
                    if ID not in ID_list:
                        ID_list.append(ID)
                        break
                ChrA = ChrB = item[0][3:]
                StartA = EndA = item[1]
                StartB = EndB = item[2]
            else: # the break do have an ID
                ID = item[0] # the ID is supposed to be the first entry
                ChrA = ChrB = item[1][3:]
                StartA = EndA = item[2]
                StartB = EndB = item[3]
            CnvType = 'DELETION'
            Orientation = 'TH'

            line = list_to_line([ID, ChrA, StartA, EndA, ChrB, StartB, EndB, CnvType, Orientation], '\t')
            outfile.write(line + '\n')
    print('OK')
# list_of_breaks = gen_rnd_breaks(how_many=100, chromosome='Y', min_distance=1000, max_distance=15000, GapTable_file='tables/gap.txt')
# gen_deletion_dataset_from_breaks(list_of_breaks, 'test_deletion_dataset.txt')
# ## Generate (m) RANDOM datasets of different length (n)
# for m in range(1000):
#   for n in [100,1000,10000,100000,1000000]:
#       outfile = 'rnd_dataset_'+ str(n)+'_'+str(m)+'.txt'
#       breaks = list()
#       for chromosome in CHROMOSOMES:      
#           breaks.extend(gen_rnd_breaks(how_many=500, chromosome=chromosome, min_distance=0, max_distance=n))
#       gen_deletion_dataset_from_breaks(breaks, outfile)

def gen_rnd_breaks(how_many=100, chromosome='Y', min_distance=1000, max_distance=15000, GapTable_file='tables/gap.txt'):
    '''Returns tuples containing 1)the chromosome, 2)first breakpoint, 3)second breakpoint
    Keeps only the points that do not appear in te gap table.
    gen_rnd_breaks(int, string, int, int, filepath) => [(chrX, int, int), ...]
    valid chromosomes inputs are "1" to "22" ; "Y" ; "X"
    The chromosome length is based on the build GRCh37/hg19.'''

    # CHR_LENGTHS is based on GRCh37
    CHR_LENGTHS = {'1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
               '5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
               '9' :141213431,'10':135534747,'11':135006516,'12':133851895,
               '13':115169878,'14':107349540,'15':102531392,'16':90354753,
               '17':81195210,'18':78077248,'19':59128983,'20':63025520,
               '21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}

    # Genrates a chromosome-specific gap list
    print('generating', how_many, 'breakpoints in Chr', chromosome, '..........')
    with open(GapTable_file,'r') as infile:
        lines = infile.readlines()
    
    full_gap_list = []
    chr_specific_gap = []
    for line in lines:
        if '#' not in line: # skip comments
            full_gap_list.append(line_to_list(line, '\t'))
    
    for item in full_gap_list:
        if 'chr' + chromosome in item:
            # Database/browser start coordinates differ by 1 base
            chr_specific_gap.append((item[2],item[3]))

    # Merge contiguous gaps
    merged_gaps = []
    n = 0
    left_tick = False
    while n < len(chr_specific_gap):
        if left_tick == False:
            left_tick = chr_specific_gap[n][0]
        try:
            if chr_specific_gap[n][1] == chr_specific_gap[n+1][0]:
                n += 1
            else:
                right_tick = chr_specific_gap[n][1]
                merged_gaps.append((left_tick,right_tick))
                left_tick = False
                n += 1
        except:
            n += 1

    # Genrates breakpoint list
    list_of_breakpoints = []
    while len(list_of_breakpoints) < how_many:
        try:
            start = random.randint(0,CHR_LENGTHS[chromosome])
        except KeyError:
            if chromosome == '23':
                chromosome = 'X'
                start = random.randint(0,CHR_LENGTHS[chromosome])
            elif chromosome == '24':
                chromosome = 'Y'
                start = random.randint(0,CHR_LENGTHS[chromosome])
            else:
                print('ERROR: Wrong chromosome name!!')


        end = random.randint(start+min_distance, start+max_distance)
        are_points_ok = True # assumes that the points are ok
        
        for item in merged_gaps:
            # checks whether the points are ok for real
            if start < int(item[0]) or start > int(item[1]):
                if end < int(item[0]) or end > int(item[1]):
                    pass
                else: are_points_ok = False
            else: are_points_ok = False

        if are_points_ok == True:           
            list_of_breakpoints.append(('chr'+chromosome, start, end))
    print('OK')
    return list_of_breakpoints
# print(gen_rnd_breaks(how_many=100, chromosome='Y', min_distance=1000, max_distance=15000, GapTable_file='tables/gap.txt'))

def gen_rnd_id(length):
    
    '''Generates a random string made by uppercase ascii chars and digits'''
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for char in range(length))
# print(gen_rnd_id(16))

#@profile
def gen_rnd_single_break(how_many=100, chromosome='1', GapTable_file='/Users/amarcozzi/Desktop/All_breakpoints_HG19_gap_10k.txt', verbose=False):
    '''Returns tuples containing 1)the chromosome, 2)the breakpoint
    Keeps only the points that do not appear in te gap table.
    gen_rnd_breaks(int, string, filepath) => [(chrX, int), ...]
    valid chromosomes inputs are "1" to "22" ; "Y" ; "X"
    The chromosome length is based on the build GRCh37/hg19.
    Prerequisites: The gap_list file is in the form:
                                                        ##chrom chromStart  chromEnd
                                                        chr1    0   10000
                                                        chr1    30000   40000
                                                        chr1    40000   50000
                                                        chr1    50000   60000
    '''
    if verbose == True:
        start_time = time.time()

    # CHR_LENGTHS is based on GRCh37
    CHR_LENGTHS = {'1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
               '5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
               '9' :141213431,'10':135534747,'11':135006516,'12':133851895,
               '13':115169878,'14':107349540,'15':102531392,'16':90354753,
               '17':81195210,'18':78077248,'19':59128983,'20':63025520,
               '21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}

    # Genrates a chromosome-specific gap list
    with open(GapTable_file, 'r') as infile:
        lines = infile.readlines()
    
    full_gap_list = []
    chr_specific_gap = []
    for line in lines:
        if '#' not in line: # skip comments
            full_gap_list.append(line_to_list(line, '\t'))
    
    for item in full_gap_list:
        if 'chr' + chromosome in item:
            chr_specific_gap.append((item[1],item[2]))

    # Merge contiguous gaps
    merged_gaps = merge_gaps(chr_specific_gap)
    # merged_gaps = []
    # while len(chr_specific_gap) > 0:
    #   try:
    #       if chr_specific_gap[0][1] == chr_specific_gap[1][0]:
    #           tmp = (chr_specific_gap[0][0],chr_specific_gap[1][1])
    #           chr_specific_gap.pop(0)
    #           chr_specific_gap[0] = tmp
    #       else:
    #           merged_gaps.append(chr_specific_gap.pop(0))
    #   except:
    #       merged_gaps.append(chr_specific_gap.pop(0))

    # Genrates breakpoint list
    if verbose == True: print('generating', how_many, 'breakpoints in Chr', chromosome)
    list_of_breakpoints = []
    while len(list_of_breakpoints) < how_many:
        try:
            start = random.randint(0,CHR_LENGTHS[chromosome])
            # if verbose == True: print(start)
        except KeyError:
            if chromosome == '23':
                chromosome = 'X'
                start = random.randint(0,CHR_LENGTHS[chromosome])
            elif chromosome == '24':
                chromosome = 'Y'
                start = random.randint(0,CHR_LENGTHS[chromosome])
            else:
                print('ERROR: Wrong chromosome name!!')

        #end = random.randint(start+min_distance, start+max_distance)
        are_points_ok = True # assumes that the points are ok
        
        for item in merged_gaps:
            # checks whether the points are ok for real
            if start <= int(item[0]) or start >= int(item[1]):
                pass
            else:
                are_points_ok = False
                if verbose == True: print(start,'is in a gap and will be discarded')
            
        if are_points_ok == True:           
            list_of_breakpoints.append((chromosome, start))
            if verbose == True: print(start,'is OK',len(list_of_breakpoints),'good breaks generated out of',how_many)

    if verbose == True: print(how_many,'breakpoint have been generated in chromosome',chromosome,'in',time.time()-start_time,'seconds')
    return list_of_breakpoints
# gen_rnd_single_break(verbose=True)
# ## Generate single breaks dataset
# import time
# start = time.time()
# breaks_on_1 = gen_rnd_single_break(how_many=19147,verbose=False)
# for item in breaks_on_1:
#   print(str(item[0])+'\t'+str(item[1]))
# print('Done in', time.time()-start,'seconds..')
# ## Generate a control file
# list_brkps = gen_rnd_single_break(how_many=20873, chromosome='1', GapTable_file='/Users/amarcozzi/Desktop/current_brkps_DB/out_ALL_gap.txt', verbose=True)
# with open('/Users/amarcozzi/Desktop/current_brkps_DB/out_chr1_control.txt','w') as f:
#   for item in list_brkps:
#       f.write(list_to_line(item,'\t')+'\n')
# ## Generate multiple controls
# import time
# from threading import Thread
# start_time = time.time()
# threads = 0
# running_threads = 0
# max_simultaneous_threads = 20
# GapTable_file = '/Users/amarcozzi/Desktop/Projects/Anne_Project/current_brkps_DB/out_ALL_gap.txt'
# chromosome = 'Y'
# infile = '/Users/amarcozzi/Desktop/Projects/Anne_Project/current_brkps_DB/out_chr'+chromosome+'.txt'
# how_many = 0
# for line in yield_file(infile):
#   if line.startswith(chromosome+'\t'):
#       how_many += 1
# print('found',how_many,'breakpoints in chromosome',chromosome)
# while threads < 100:
#   while running_threads >= max_simultaneous_threads:
#       time.sleep(1)
#   running_threads += 1
#   outfile = '/Users/amarcozzi/Desktop/Projects/Anne_Project/current_brkps_DB/controls/out_chr'+chromosome+'_control_'+str(threads)+'.txt'
#   print('thread', threads, '|', 'running threads:',running_threads)
#   Thread(target=gen_controls, args=(how_many,chromosome,GapTable_file,outfile)).start()
#   threads += 1
# print('Waiting for threads to finish...')
# while running_threads > 0:
#   time.sleep(1)
# end_time = time.time()
# print('\nDone in',(end_time-start_time)/60,'minutes')

def kmers_finder(sequence_dict, motif_length, min_repetition):
    '''(dict, int, int) => OrderedDict(sorted(list))
    Find all the motifs long 'motif_length' and repeated at least 'min_repetition' times.
    Return an OrderedDict having motif:repetition as key:value sorted by value. 
    '''
    motif_dict = {}
    for _id, sequence in sequence_dict.items():
        #populate a dictionary of motifs (motif_dict)
        for i in range(len(sequence) - motif_length +1):
            motif = sequence[i:i+motif_length]
            if motif not in motif_dict:
                motif_dict[motif] = 1
            else:
                motif_dict[motif] += 1

    #remove from motif_dict all the motifs repeated less than 'repetition' times
    keys_to_remove = [key for key, value in motif_dict.items() if value < min_repetition]
    for key in keys_to_remove:
        del motif_dict[key]
    
    #Return a sorted dictionary
    return OrderedDict(sorted(motif_dict.items(), key=itemgetter(1), reverse=True))

def kmers_finder_with_mismatches(sequence, motif_length, max_mismatches, most_common=False):
    '''(str, int, int) => sorted(list)
    Find the most frequent k-mers with mismatches in a string.
    Input: A sequence and a pair of integers: motif_length (<=12) and max_mismatch (<= 3).
    Output: An OrderedDict containing all k-mers with up to d mismatches in string.
    Sample Input:   ACGTTGCATGTCGCATGATGCATGAGAGCT 4 1
    Sample Output:  OrderedDict([('ATGC', 5), ('ATGT', 5), ('GATG', 5),...])
    '''


    #check passed variables
    if not motif_length <= 12 and motif_length >= 1:
        raise ValueError("motif_length must be between 0 and 12. {} was passed.".format(motif_length))
    if not max_mismatches <= 3 and max_mismatches >= 1:
        raise ValueError("max_mismatch must be between 0 and 3. {} was passed.".format(max_mismatches))

    motif_dict = {}
    for i in range(len(sequence) - motif_length +1):
        motif = sequence[i:i+motif_length]
        if motif not in motif_dict:
            motif_dict[motif] = 1
        else:
            motif_dict[motif] += 1

    motif_dict_with_mismatches = {}
    for kmer in motif_dict:
        motif_dict_with_mismatches.update({kmer:[]})
        for other_kmer in motif_dict:
            mismatches = 0
            for i in range(len(kmer)):
                if kmer[i] != other_kmer[i]:
                    mismatches += 1
            if mismatches <= max_mismatches:
                motif_dict_with_mismatches[kmer].append([other_kmer,motif_dict[other_kmer]])

    tmp = {}
    for item in motif_dict_with_mismatches:
        count = 0
        for motif in motif_dict_with_mismatches[item]:
            count += motif[-1]
        tmp.update({item:count})

    result = OrderedDict(sorted(tmp.items(), key=itemgetter(1), reverse=True))
    
    if most_common:
        commons = OrderedDict()
        _max = result.items()[0][1]
        for item in result:
            if result[item] == _max:
                commons.update({item:result[item]})
            else:
                return commons

    return result

def line_to_list(line, char):
    '''Makes a list of string out of a line. Splits the word at char.'''
    # Allows for more customization compared with string.split()
    split_indexes = find(line, char)
    list_ = []
    n = 0
    for index in split_indexes:
        item = line[n:index].replace('\n','').replace('\r','') # cleans up the line
        if item != '': # skips empty 'cells'
            list_.append(item)
        n = index + 1
    list_.append(line[n:].replace('\n','').replace('\r','')) # append the last item 
    return list_
# print(line_to_list('Makes a list of string out of a line. Splits the word at char.', ' '))

def list_to_line(list_, char):
    '''Makes a string out of a list of items'''
    # Allows for more customization compared with string.split()
    string = ''
    for item in list_:
        string += str(item) + char
    return string.rstrip(char) # Removes the last char
#print(list_to_line(['prova', '1', '2', '3', 'prova'], '---'))

def list_of_files(path, extension, recursive=False):
    '''
    Return a list of filepaths for each file into path with the target extension.
    If recursive, it will loop over subfolders as well.
    '''
    if not recursive:
        for file_path in glob.iglob(path + '/*.' + extension):
            yield file_path
    else:
        for root, dirs, files in os.walk(path):
            for file_path in glob.iglob(root + '/*.' + extension):
                yield file_path

def merge_gaps(gap_list):
    '''
    Merges overlapping gaps in a gap list.
    The gap list is in the form: [('3','4'),('5','6'),('6','7'),('8','9'),('10','11'),('15','16'),('17','18'),('18','19')]
    Returns a new list containing the merged gaps: [('3','4'),('5','7'),('8','9'),('10','11'),('15','16'),('17','19')]
    '''
    merged_gaps = []
    while len(gap_list) > 0:
        try:
            if int(gap_list[0][1]) >= int(gap_list[1][0]):
                tmp = (gap_list[0][0],gap_list[1][1])
                gap_list.pop(0)
                gap_list[0] = tmp
            else:
                merged_gaps.append(gap_list.pop(0))
        except:
            merged_gaps.append(gap_list.pop(0))
    return merged_gaps
# gap_list = [('3','4'),('5','6'),('6','7'),('8','9'),('10','11'),('15','16'),('17','18'),('18','19')]
# expected = [('3','4'),('5','7'),('8','9'),('10','11'),('15','16'),('17','19')]
# prova = merge_gaps(gap_list)
# print(prova)
# print(expected)

def merge_sort(intervals):
    '''
    Merges and sorts the intervals in a list.
    It's an alternative of merge_gaps() that sort the list before merging.
    Should be faster but I haven't campared them yet.
    '''
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged 

def multi_threads_fusion_genes_annotation(folder_path, extension, max_simultaneous_threads):
    ''' Executes annotate_fusion_genes() for each dataset file in a folder.
    Each execution run on a different thread.'''
    global running_threads
    dataset_files = list_of_files(folder_path, extension)
    threads = 0
    running_threads = 0
    for file_ in dataset_files:
        while running_threads >= max_simultaneous_threads:
            time.sleep(1)
        threads += 1
        running_threads += 1
        print('thread', threads, '|', 'running threads:',running_threads)
        Thread(target=annotate_fusion_genes, args=(file_,)).start() # with multithreading
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public'
# multi_threads_fusion_genes_annotation(folder, 'txt',50)

def pandize_dataset(annotated_dataset, verbose=True):
    '''
    Prepares a dataset to be "pandas ready".
    Takes a file path as input.
    '''
    # Parse
    if verbose == True:
        message = 'parsing ' + annotated_dataset.split('/')[-1]
        spacer = (100-len(message))*'.'
        print(message, spacer)

    dataset = pd.io.parsers.read_table(annotated_dataset, dtype={'ChrA':'str','ChrB':'str'}, sep='\t', index_col=0)
    if verbose == True:
        print('OK')
    
    # Clean
    if verbose == True:
        message = 'cleaning ' + annotated_dataset.split('/')[-1]
        spacer = (100-len(message))*'.'
        print(message, spacer)

    dataset = dataset.replace('In Frame', 1)
    dataset = dataset.replace('Not in Frame', 0)
    dataset = dataset.replace('In Phase', 1)
    dataset = dataset.replace('Not in Phase', 0)
    if verbose == True:
        print('OK')

    return dataset
# pandize_dataset('test_data_annotated.txt')
# pandize_dataset('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/control_dataset_100-1000-150000_annotated.txt')

def parse_blastXML(infile):
    '''
    Parses a blast outfile (XML).
    '''
    
    for blast_record in NCBIXML.parse(open(infile)):
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                print("*****Alignment****")
                print("sequence:", alignment.title)
                print("length:", alignment.length)
                print("e-value:", hsp.expect)
                print(hsp.query)
                print(hsp.match)
                print(hsp.sbjct)
# to be tested

def reverse(sequence):
    r = ''
    for i in range(len(sequence),0,-1):
        r += sequence[i-1]
    return r

def complement(sequence):
    d = {'A':'T','a':'t',
         'T':'A','t':'a',
         'C':'G','c':'g',
         'G':'C','g':'c'}
    r = ''
    for b in sequence.upper():
        r += d[b]
    return r


def get_mismatches(template, primer, maxerr, overlapped=False):
    error = 'e<={}'.format(maxerr)
    return regex.findall(f'({primer}){{{error}}}', template, overlapped=overlapped)


def pcr(template,primer_F,primer_R,circular=False):
    if circular: ##works only with primers without 5' overhang
        i = template.upper().find(primer_F.upper())
        template = template[i:]+template[:i]
    
    #Find primer_F, or the largest 3'part of it, in template
    for n in range(len(primer_F)):
        ix_F = [m.end() for m in re.finditer(primer_F[n:].upper(),
                                           template.upper())]
        if len(ix_F) == 1: #it's unique
            #print(ix_F)
            #print(primer_F[n:])
            break
        n += 1
    #print(ix_F)
    #Find primer_R, or the largest 5'part of it, in template
    rc_R = reverse(complement(primer_R))
    for n in range(len(primer_R)):
        ix = [m.start() for m in re.finditer(rc_R[:n].upper(),
                                           template.upper())]
        if len(ix) == 1: #it's unique
            ix_R = ix[:]

        if len(ix) < 1: #it's the largest possible
            #print(ix_R)
            #print(rc_R[:n])
            break
        n += 1
    #Build the product
    return primer_F + template[ix_F[0]:ix_R[0]] + rc_R
##template = 'CTAGAGAGGGCCTATTTCCCATGATT--something--GCCAATTCTGCAGACAAATGGGGTACCCG'
##primer_F = 'GACAAATGGCTCTAGAGAGGGCCTATTTCCCATGATT'
##primer_R = 'TTATGTAACGGGTACCCCATTTGTCTGCAGAATTGGC'
##product = pcr(template,primer_F,primer_R)
##expected = 'GACAAATGGCTCTAGAGAGGGCCTATTTCCCATGATT--something--GCCAATTCTGCAGACAAATGGGGTACCCGTTACATAA'
##expected == result

def pip_upgrade_all(executable=False):
    '''
    Upgrades all pip-installed packages.
    Requires a bash shell.
    '''
    if executable:
        print('upgrading pip...')
        call(f'{executable} -m pip install --upgrade pip',
             shell=True)
        call(f"{executable} -m pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 {executable} -m pip install -U",
             shell=True)
        print('done')
        
    else:
        #pip
        print('upgrading pip...')
        call('python -m pip install --upgrade pip', shell=True)
        call("python -m pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 python -m pip install -U", shell=True)
        #pip2
        print('upgrading pip2...')
        call('python2 -m pip install --upgrade pip', shell=True)
        call("python2 -m pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 python2 -m pip install -U", shell=True)
        #pip3
        print('upgrading pip3...')
        call('python3 -m pip install --upgrade pip', shell=True)
        call("python3 -m pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 python3 -m pip install -U", shell=True)
        #pypy
        print('upgrading pypy-pip...')
        call('pypy -m pip install --upgrade pip',shell=True)
        call("pypy -m pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pypy -m pip install -U", shell=True)
        #pypy3
        print('upgrading pypy3-pip...')
        call('pypy3 -m pip install --upgrade pip',shell=True)
        call("pypy3 -m pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pypy3 -m pip install -U", shell=True)

def probability(p,n,k):
    '''
    Simple probability calculator.
    Calculates what is the probability that k events occur in n trials.
    Each event have p probability of occurring once.
    Example: What is the probability of having 3 Heads by flipping a coin 10 times?
    probability = prob(0.5,10,3)
    print(probability) => (15/128) = 0.1171875
    '''

    p = float(p)
    n = float(n)
    k = float(k)
    C = math.factorial(n) / (math.factorial(k) * math.factorial(n-k) )
    probability = C * (p**k) * (1-p)**(n-k)
    return probability
#from math import factorial
#print(probability(0.5,10,3))
#print(probability(0.5,1,1))

def process(real_dataset):
    '''
    Generates, annotates and sorts a controll dataset for the given real dataset.
    '''
    gen_control_dataset(real_dataset)
    control_filename = real_dataset[:-4]+'_control.txt'

    #annotate_fusion_genes(real_dataset)
    annotate_fusion_genes(control_filename)

    control_filename = control_filename[:-4]+'_annotated.txt'
    #dataset_filename = real_dataset[:-4]+'_annotated.txt'
    
    #sort_dataset(dataset_filename)
    sort_dataset(control_filename)

    print(real_dataset,'processed. All OK.')
#process('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/clinvarCnv-DeletionsOnly.txt')
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/random'
# for item in list_of_files(folder,'txt'):
#   process(item)

def query_encode(chromosome, start, end):
    '''
    Queries ENCODE via http://promoter.bx.psu.edu/ENCODE/search_human.php
    Parses the output and returns a dictionary of CIS elements found and the relative location.
    '''

    ## Regex setup
    re1='(chr{})'.format(chromosome) # The specific chromosome
    re2='(:)'    # Any Single Character ':'
    re3='(\\d+)' # Integer
    re4='(-)'    # Any Single Character '-'
    re5='(\\d+)' # Integer
    rg = re.compile(re1+re2+re3+re4+re5,re.IGNORECASE|re.DOTALL)

    ## Query ENCODE
    std_link = 'http://promoter.bx.psu.edu/ENCODE/get_human_cis_region.php?assembly=hg19&'
    query = std_link + 'chr=chr{}&start={}&end={}'.format(chromosome,start,end)
    print(query)
    html_doc = urlopen(query)
    html_txt = BeautifulSoup(html_doc, 'html.parser').get_text()
    data = html_txt.split('\n')

    ## Parse the output
    parsed = {}
    coordinates = [i for i, item_ in enumerate(data) if item_.strip() == 'Coordinate']
    elements = [data[i-2].split('  ')[-1].replace(': ','') for i in coordinates]
    blocks = [item for item in data if item[:3] == 'chr']
    #if len(elements) == len(blocks):
    i = 0
    for item in elements:
        txt = blocks[i]
        m = rg.findall(txt)
        bins = [''.join(item) for item in m]
        parsed.update({item:bins})
        i += 1
            
    return parsed
#cis_elements = query_encode(2,10000,20000)

def run_perl(perl_script_file, input_perl_script):
    '''
    Run an external perl script and return its output
    '''
    
    return check_output(["perl", perl_script_file, input_perl_script])
#print(run_perl('FusionGenes_Annotation.pl', 'test_data.txt'))

def run_py(code, interp='python3'):
    '''Run an block of python code using the target interpreter.'''
    with open('tmp.py', 'w') as f:
        for line in code.split('\n'):
            f.write(line+'\n')
    return check_output([interpr, 'tmp.py'])

def run_pypy(code, interpr='pypy3'):
    '''Run an block of python code with PyPy'''
    with open('tmp.py', 'w') as f:
        for line in code.split('\n'):
            f.write(line+'\n')
    return check_output([interpr, 'tmp.py'])

def sequence_from_coordinates(chromosome,strand,start,end): #beta hg19 only
    '''
    Download the nucleotide sequence from the gene_name.
    '''
    Entrez.email = "a.marcozzi@umcutrecht.nl" # Always tell NCBI who you are
    
    #GRCh37 from http://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.25/#/def_asm_Primary_Assembly
    NCBI_IDS = {'1':'NC_000001.10','2':'NC_000002.11','3':'NC_000003.11','4':'NC_000004.11',
                '5':'NC_000005.9','6':'NC_000006.11','7':'NC_000007.13','8':'NC_000008.10',
                '9':'NC_000009.11','10':'NC_000010.10','11':'NC_000011.9','12':'NC_000012.11',
                '13':'NC_000013.10','14':'NC_000014.8','15':'NC_000015.9','16':'NC_000016.9',
                '17':'NC_000017.10','18':'NC_000018.9','19':'NC_000019.9','20':'NC_000020.10',
                '21':'NC_000021.8','22':'NC_000022.10','X':'NC_000023.10','Y':'NC_000024.9'}       
  
    try:        
        handle = Entrez.efetch(db="nucleotide", 
                               id=NCBI_IDS[str(chromosome)], 
                               rettype="fasta", 
                               strand=strand, #"1" for the plus strand and "2" for the minus strand.
                               seq_start=start,
                               seq_stop=end)
        record = SeqIO.read(handle, "fasta")
        handle.close()
        sequence = str(record.seq)
        return sequence
    except ValueError:
        print('ValueError: no sequence found in NCBI')
        return False
#a = sequence_from_coordinates(9,'-',21967751,21994490)
#print(a)

def sequence_from_gene(gene_name): #beta
    '''
    Download the nucleotide sequence from the gene_name.
    '''
    data = EnsemblRelease(75)
    Entrez.email = "a.marcozzi@umcutrecht.nl" # Always tell NCBI who you are
    NCBI_IDS = {'1':"NC_000001", '2':"NC_000002",'3':"NC_000003",'4':"NC_000004",
                '5':"NC_000005",'6':"NC_000006",'7':"NC_000007", '8':"NC_000008",
                '9':"NC_000009", '10':"NC_000010", '11':"NC_000011", '12':"NC_000012",
                '13':"NC_000013",'14':"NC_000014", '15':"NC_000015", '16':"NC_000016", 
                '17':"NC_000017", '18':"NC_000018", '19':"NC_000019", '20':"NC_000020",
                '21':"NC_000021", '22':"NC_000022", 'X':"NC_000023", 'Y':"NC_000024"}
    
    gene_obj = data.genes_by_name(gene_name)
    target_chromosome = NCBI_IDS[gene_obj[0].contig]
    seq_start = int(gene_obj[0].start)
    seq_stop = int(gene_obj[0].end)
    strand = 1 if gene_obj[0].strand == '+' else 2
        
    try:
             
        handle = Entrez.efetch(db="nucleotide", 
                               id=target_chromosome, 
                               rettype="fasta", 
                               strand=strand, #"1" for the plus strand and "2" for the minus strand.
                               seq_start=seq_start,
                               seq_stop=seq_stop)
        record = SeqIO.read(handle, "fasta")
        handle.close()
        sequence = str(record.seq)
        return sequence

    except ValueError:
        print('ValueError: no sequence found in NCBI')
        return False

def sortby_chr(string):
    '''
    Helps to sort datasets grouped by ChrA/B.
    To use with sorted().
    '''
    # since the ChrA/B value is a string, when sorting by chr may return ['1','10','11'...'2','20'...'3'...'X','Y']
    # instead I want sorted() to return ['1','2',...'9','10','11'...'X','Y']
    if string == 'X':
        return 23
    elif string == 'Y':
        return 24
    else:
        return int(string)
# prova = ['1','10','11','9','2','20','3','X','Y']
# print('sorted()', sorted(prova))
# print('sortby_chr()', sorted(prova, key=sortby_chr))

def sort_dataset(dataset_file, overwrite=False):
    '''
    Sort a dataset by ChrA. It helps during plotting
    '''
    text = []
    header_counter = 0
    header = False
    print('Sorting...')
    with open(dataset_file, 'r') as infile:
        #lines = infile.readlines()
        for line in infile:
            list_ = line_to_list(line, '\t')
            if line[:2] == '##':
                header = list_
                header_counter += 1
            else:
                text.append(list_)
    #checkpoint
    if header == False or header_counter > 1:
        print('Something is wrong with the header line...', header_counter, header)
        return None     
    # sort by the second element of the list i.e. 'ChrA'
    text.sort(key=lambda x: sortby_chr(itemgetter(1)(x))) 
    # Write output
    if overwrite == False:
        outfile = dataset_file[:-4]+'_sorted.txt'
    else:
        outfile = dataset_files
    with open(outfile, 'w') as outfile:
        outfile.write(list_to_line(header, '\t') + '\n')
        for list_ in text:
            outfile.write(list_to_line(list_, '\t') + '\n')
    print('Done!')
# sort_dataset('test_data.txt')
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public'
# for item in list_of_files(folder, 'txt'):
#   sort_dataset(item)
# sort_dataset('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/annotated/dgvMerged-DeletionsOnly_annotated.txt')

def split_fasta_file(infile): #beta
    '''
    Split a fasta file containing multiple sequences
    into multiple files containing one sequence each.
    One sequence per file.
    '''
    flag = False
    length = 0
    with open(infile,'r') as f:
        for line in f:
            if line.startswith('>'):
                if flag == False:
                    flag = True
                    outfile = '{}.txt'.format(line[1:].strip())
                    print('writing {}'.format(outfile))
                    lines = [line]
                else:
                    with open(outfile, 'w') as out:
                        for _ in lines:
                            out.write(_)
                        print('{} bases written'.format(length))
                        length = 0

                    outfile = '{}.txt'.format(line[1:].strip())
                    print('writing {}'.format(outfile))
                    lines = [line]
            else:
                lines.append(line)
                length += len(line.strip())

        #Write last file        
        with open(outfile, 'w') as out:
            for _ in lines:
                out.write(_)
        print('{} bases written'.format(length))    

def substract_datasets(infile_1, infile_2, outfile, header=True):
    '''
    Takes two files containing tab delimited data, comapares them and return a file
    containing the data that is present only in infile_2 but not in infile_1.
    The variable by_column is an int that indicates which column to use
    as data reference for the comparison.
    '''
    header2 = False
    comment_line = '# dataset generated by substracting ' + infile_1 + ' to ' + infile_2 + '\n'

    with open(infile_1) as infile_1:
        lines_1 = infile_1.readlines()
    with open(infile_2) as infile_2:
        lines_2 = infile_2.readlines()

    row_to_removes = []
    for line in lines_1:
        if line[0] != '#': # skips comments
            if header == True:
                header2 = True # to use for the second file
                header = False # set back header to false since the first line will be skipped
                first_line = line
                pass
            else:
                item = line_to_list(line, '\t')
                row_to_removes.append(item)

    result_list = []
    for line in lines_2:
        if line[0] != '#': # skips comments
            if header2 == True:
                header2 = False # set back header to false since the first line will be skipped
                pass
            else:
                item = line_to_list(line, '\t')
                if item not in row_to_removes:
                    result_list.append(item)
    
    with open(outfile, 'w') as outfile:
        outfile.write(comment_line)
        outfile.write(first_line)
        for item in result_list:
            outfile.write(list_to_line(item, '\t') + '\n')
    print('substraction of two datasets DONE')
# substract_datasets('dataset_1_b.txt', 'dataset_1.txt', 'dataset_1-1b.txt', header=True)

def yield_file(filepath):
    '''
    A simple generator that yield the lines of a file.
    Good to read large file without running out of memory.
    '''
    with open(filepath, 'r') as f:
        for line in f:
            yield line
# for line in yield_file('GRCh37_hg19_variants_2014-10-16.txt'):
#   print(line[:20])

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data