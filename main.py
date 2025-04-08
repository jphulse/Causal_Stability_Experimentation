# This code was created for my semester research paper for CSC 591 at NCSU in Fall 2024
# The goal is to try to improve stability seen in causal discovery by using subsamples of synthetic
# and real data in order to predict the most stable algorithm to use on a dataset.  This was primarily inspired 
# by a paper published in 2019 C. Glymour, K. Zhang, and P. Spirtes, ‘‘Review of causal discovery meth-
# ods based on graphical models,’’ Frontiers in genetics, vol. 10, p. 524, 2019.
# where they discuss assumptions inherit in the models, and briefly discuss instability.
# That prompted me to look into dataset attributes and see if there was a way to use the existing innovations
# in causal discovery to make stability-maximizing predictions without needing to "reinvent the wheel" for every new dataset
#
#
# @author Jeremy P. Hulse CSC Undergraduate and Accelerated Master's Student at North Carolina State University

# General imports for files or system operations
import os 
import re
import sys 
import statistics
# pip install portalocker
import portalocker
from typing import List

# Data formattingimports
import pandas as pd
import numpy as np

from pandas import DataFrame
from numpy import ndarray
# pip install causallearn
# causal learning imports
# Causallearn is a novel package for causal discovery associated with a 2024
# paper which is already being cited within other new literature, already having 47 citations 
# in the field make this a new representation of the latest and greatest in causal discovery
# and the 30 years of work that have gone into improving these methods
from causallearn.search.ConstraintBased import PC
from causallearn.search.ConstraintBased import FCI
from causallearn.search.ScoreBased import GES
from causallearn.search.FCMBased import lingam
# Not currently used, but provides utilities to draw the graphs and other fun stuff to play with, so I left it in
from causallearn.utils.GraphUtils import GraphUtils

# data processing or modification imports
from scipy import stats
import statsmodels.api as sm
import bisect

# class representing shared DATA attributes in order to split the
# functionality of experiments while complying with DRY principle
class DATA:
    # Sets fields initializing object
    def __init__(self, file):
        self.file = file
        self.pc = 0.0 # average PC score, max 1, min 0
        self.fci = 0.0 # average FCI score, max 1, min 0
        self.ges = 0.0 # average GES score, max 1, min 0
        self.lin = 0.0 # average LiNGAM score, max 1, min 0
        self.count = 0 # count of how many scores we've read
        self.pc_graphs = []
        self.fci_graphs = []
        self.ges_graphs = []
        self.lin_graphs =[]
        self.prop_pc = None
        self.prop_fci = None
        self.prop_ges = None
        self.prop_lin = None
        self.potential_conf = False
        self.actuals = []
        self.worsts = []

    # we made a type of DATA so it is neither SYNTHETIC or REAL
    # here to support overriding methods in subclasses for compiler
    def type(self):
        return 'unknown'
    
    # Add an observation, we have recorded new Jaccard indexes for each of the algs on the dataset within file
    def addObservation(self, pc, fci, ges, lin,  first=False):
        if first:
            self.pc = 0
            self.lin = 0
            self.ges = 0
            self.fci = 0
        self.pc = (self.pc * self.count + pc) / (self.count + 1)
        self.fci = (self.fci * self.count + fci) / (self.count + 1)
        self.ges = (self.ges * self.count + ges) / (self.count + 1)
        self.lin = (self.lin * self.count + lin) / (self.count + 1)
        self.count += 1

    # Reports the RQ1 results associated with this data, the RQ2 results are reported in the main method and are associated with the 
    # PROPERTY object that we maintain in execution.  The format of output for now
    # is as follows
    # <input file> <N (iterations)> <average pc score> <average fci score> <average ges score> <average LiNGAM score>
    def reportRQ1(self):
        with open('results/RQ1.csv', 'a+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                print(self.pc, self.fci, self.ges, self.lin, sep=',', file=f, end='\n' )
            finally:
                portalocker.unlock(f)
    
    def reportRQ3(self):
        with open('results/RQ3.csv', 'a+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                print(self.pc, self.fci, self.ges, self.lin, sep=',', file=f, end='\n' )
            finally:
                portalocker.unlock(f)
    def reportRQ7(self):
        with open('results/RQ7.csv', 'a+') as f:
             portalocker.lock(f, portalocker.LOCK_EX)
             try:
                for i, j in zip(self.actuals, self.worsts):
                    print(i, j, sep=',', file=f,end='\n' )
                    
                    
             finally:
                portalocker.unlock(f)

    def reportRQ8(self):
        with open('results/RQ8.csv', 'a+') as f:
             portalocker.lock(f, portalocker.LOCK_EX)
             try:
                for i, j in zip(self.actuals, self.worsts):
                    print(i, j, sep=',', file=f,end='\n' )
                    
                    
             finally:
                portalocker.unlock(f)
    # Getter methods
    def getFile(self):
        return self.file
    
    def getPC(self): 
        return self.pc
    
    def getFCI(self):
        return self.fci
    
    def getGES(self):
        return self.ges
    
    def getLiNGAM(self):
        return self.lin
    
    def getCount(self):
        return self.count
    
    # compares the input algorithm to the bets performance seen by the
    # DATA object to this point during experiments, returns true if the alg
    # is equal to the max algorithm, or if the absolute value of the difference 
    # in performance is within 5 % of the real "best" algorithms performance
    # Essentially saying that the alg we selected is likely "close" to the best choice
    # for this DATA
    def compareToMax(self, alg):
        exp = self.pc
        if alg ==2:
            exp = self.fci
        if alg == 3:
            exp = self.ges
        if alg == 4:
            exp = self.lin
        ac, ac_value = most_stable_alg(self)
        return ac == alg or abs(ac_value - exp) < (ac_value * .05)
    
    # Gets the performance in Jaccard index of the worst performing algorithm associated with this DATA
    def getWorst(self):
        exp = self.pc 
        if exp > self.fci:
            exp = self.fci
        if exp > self.ges:
            exp = self.ges
        if exp > self.lin:
            exp = self.lin
        return exp
    
    # Gets the expected performance of the given alg, 1 = PC , 2 = FCI, 3 = GES, 4 = LiNGAM
    def getAlgPerf(self, alg):
        exp = self.pc
        if alg ==2:
            exp = self.fci
        if alg == 3:
            exp = self.ges
        if alg == 4:
            exp = self.lin
        return exp
    
    def getImprovement(self, alg):
        actual = self.getAlgPerf(alg)
        self.actuals.append(actual)
        worst = self.getWorst()
        self.worsts.append(worst)
        diff = actual - worst 
        return diff / worst if worst > 0 else diff


        

# This class handles data needed to make predictions such as information
# on every subsample, without keeping the subsamples themselves
class PROPERTY:
    def __init__(self):
        self.props = [] # list of tuples of dataset info
        self.heuristic_conf = False 
        self.correct = 0 # how many we guessed correct of SYNTHETICs
        self.total = 0 # how many we guessed total for SYNTHETICs
        self.close = 0 # how many we guessed close for SYNTHETICs
        self.temp_real = [] # temporary list of unlabeled Y val tuples for REAL datasets
        self.close_real = 0 #how many we guessed close for REAL
        self.correct_real = 0 # how many we guessed correct for REAL
        self.total_real = 0 # how many total guesses made for REAL
        self.imp = 0 # Average improvement for guesses made on synth
        self.imp_real = 0 # Average improvement for guesses made on REAL

    # Preparation for the next cycle of the REAL experiments RQ3 and RQ4    
    def prep_for_second(self):
        self.temp_real = []
    
    # Reports the results of the 3rd research question into a file
    def reportRQ2(self):
        with open('results/RQ2.csv', 'a+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                print(self.correct / self.total, self.close / self.total, sep=',', file=f, end='\n' )
            finally:
                portalocker.unlock(f)
    def reportRQ4(self):
        with open('results/RQ4.csv', 'a+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                print(self.correct_real / self.total_real, self.close_real / self.total_real, sep=',', file=f, end='\n' )
            finally:
                portalocker.unlock(f)

    def reportRQ5(self):
        self.imp /= self.total
        with open('results/RQ5.csv', 'a+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                print(self.imp, sep=',', file=f, end='\n'  )
            finally:
                portalocker.unlock(f)
    def reportRQ6(self):
        self.imp_real /= self.total_real
        with open('results/RQ6.csv', 'a+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                print(self.imp_real, sep=',', file=f, end='\n'  )
            finally:
                portalocker.unlock(f)


    
# class representing specific operations to be performed on synthetic data
class SYNTHETIC(DATA):

    def type(self):
        return 'S'

# Class representing specific operations to be performed on real datasets
class REAL(DATA):
    def __init__(self, file):
        super().__init__(file)
        self.subsamples = []
        
    def type(self):
        return 'R'  





    

# INITIALIZATION, done first, regex and file info, to add more files SYNTHETIC data (exposed to RQ1 and RQ2)
# just simply place a file named train.csv in a new subdirectory from the data\\ directory.
# To add a new set of associated REAL datasets you can eithe rplace them within the specified subdirectories below,
# however I would suggest making a new folder data\\<my_new_folder> and placing them within that folder as *.csv files
# then add the path of that directory with the same syntax as the ones in the real_paths list manually
initial_path = 'data\\'
proc_paths = ['data\\Config', 'data\\Process']
real_paths = [ 'data\\ant', 'data\\ivy', 'data\\camel', 'data\\synapse', 'data\\xerces']
real_pattern = r'.*\.csv$'
synethetic_pattern = r'train.csv$'

# gets all of the files from the directory that match the provided regex pattern
# optional found flags allows for separate search, see main code at the bottom for an example of how to use this
def get_files(directory, regex, found=[], synthetic=True):
    reg = re.compile(regex)
    for path, _, files in os.walk(directory):
        for file in files:
            if reg.search(file):
                found.append(os.path.join(path, file))
    
    return found

# Takes a 2D edgelist and turns them into an adjacency matrix for comparisons
# used for the graphs output by the pc algorithm
def turnEdgesToMatrix(edges, dim):
    matrix = np.zeros((dim, dim))
    for i, j in edges:
        matrix[i, j] = 1
    
    return matrix

# Turns an edgelist  from the fci part of causallearn into an adjacency matrix
# IMPORTANT only works on data labeled X1, X2, X3 ... XN, intended for use with fci output
def turnEdgeListToMatrix(list, dim):
    edges = []
    for edge in list:
        e1 = edge.get_numerical_endpoint1()
        e2 = edge.get_numerical_endpoint2()
        if e1 == -1:
            edges.append((int(edge.get_node1().get_name()[1:]) - 1, int(edge.get_node2().get_name()[1:]) - 1))
        elif (e1 == 2 and e2 != 1) or (e1 == 1 and e2 == 1):
            edges.append((int(edge.get_node1().get_name()[1:]) - 1, int(edge.get_node2().get_name()[1:]) - 1))
    
    return turnEdgesToMatrix(edges, dim)

# GES adjacency matrix process is slightly different from the others and done here
# this is done this way currently, although edges where there is a -1 are less
# certain, treated as undirected the same way they are handles in PC by converting them to 
# 1 on both sides, violates the DAG for the sake of consistency and fairness
def getGESAdjacencyMatrix(graph, dim):
    for i in range(dim):
        for j in range(dim):
            if graph[i, j] == -1:
                graph[i, j] = 1
    
    return graph

# Adjusts the LiNGAM matrix to fit our model for evaluation, there is potential to add an additional parameter to require a certain minimal threshold effect
# to be considered an edge, this would likely improve stability at the cost of edgecount but may not be helpful in accuracy
# potential area for hyperparameter optimization for stability
def modifyLinGamAdjacencyMatrix(mat, dim):
    for i in range(dim):
        for j in range(dim):
            if mat[i, j] != 0:
                mat[i, j] = 1
    return mat

# Calculates the Jaccard Index of two adjacency matrix causal graphs
# of size dim, the equation for this is the edges in g1 AND g2 over the edges
# in g1 OR g2, and this will be a value between 0 (least similar) and 1 (most similar)
def calculateJaccardIndex(g1, g2, dim):
    bothCount = 0
    eitherCount = 0
    for i in range(dim):
        for j in range(dim):
            if g1[i, j] == 1 or g2[i, j] == 1:
                eitherCount += 1
                if g1[i, j] == g2[i, j]:
                    bothCount += 1
    return bothCount / eitherCount if eitherCount > 0 else 0

# performs the pc algorithm and returns the appropriately formatted matrix                
def performPC(grid, dim):
    pc_graph = PC.pc(grid, show_progress=False)
    return turnEdgesToMatrix(pc_graph.find_adj(), dim)

# performs the FCI alg and returns the formatted matrix
def performFCI(grid, dim):
    _, fci_list = FCI.fci(grid, show_progress=False)
    return turnEdgeListToMatrix(fci_list, dim)

# performs the GES alg and returns the formatted matrix
def performGES(grid, dim):
    ges_graph = GES.ges(grid)
    return getGESAdjacencyMatrix(ges_graph['G'].graph, dim)

# performs the LiNGAM alg and returns the formatted matrix
def performLiNGAM(grid, dim):
    lin_model = lingam.DirectLiNGAM()
    lin_model.fit(grid)
    return modifyLinGamAdjacencyMatrix(lin_model.adjacency_matrix_, dim)

# Generates an expected graph from the list of graphs with a minimum occurance rate
# to include it in the expected default .5 to enforce a majority
def generateExpectedGraph(list, dim, p=.5):
    expected = np.zeros((dim, dim))
    for matrix in list:
        for i in range(dim):
            for j in range(dim):
                expected[i, j] += matrix[i, j]
    for i in range(dim):
        for j in range(dim):
            expected[i, j] = 1 if expected[i, j] / len(list) > p else 0

    return expected 

# Calculates the jaccard index for a list of graphs compared to one expected graph
def listJaccards(list, exp, dim):
    average = 0
    
    for graph in list:
        average += calculateJaccardIndex(graph, exp, dim)
    return average / len(list)

# Performs the algorithm for RQ1:
# Goes through the dataset, and compares it with a canonical distribution if one is provided
# however, this result is messy and difficult to interpret so in the main code this is currently overwritten
# by the code within exp1Expected in order to better gauge "stability" without introducing threats to validity
# based on additional assumptions about unfamiliar datasets, but feel free to use this method externally
def exp1(grid, data: SYNTHETIC, canon):
    # Obtain graphs and adj matrices
    dim = grid.shape[1]
    pc_matrix = performPC(grid, dim)
    fci_matrix = performFCI(grid, dim)
    ges_matrix = performGES(grid, dim)
    lin_matrix = performLiNGAM(grid, dim)
    pc_jaccard = calculateJaccardIndex(pc_matrix, canon, dim)
    fci_jaccard = calculateJaccardIndex(fci_matrix, canon, dim)
    ges_jaccard = calculateJaccardIndex(ges_matrix, canon, dim)
    lin_jaccard = calculateJaccardIndex(lin_matrix, canon, dim)
    data.addObservation(pc_jaccard, fci_jaccard, ges_jaccard, lin_jaccard, data.getCount() == 0)
    data.pc_graphs.append(pc_matrix)
    data.fci_graphs.append(fci_matrix)
    data.ges_graphs.append(ges_matrix)
    data.lin_graphs.append(lin_matrix)

# Makes the four kinds of graphs for the REAl data given the 
# grid representation of the dataframe associated with the sample
def makeGraphs(grid, data: REAL):
    dim = grid.shape[1]
    pc_matrix = performPC(grid, dim)
    fci_matrix = performFCI(grid, dim)
    ges_matrix = performGES(grid, dim)
    lin_matrix = performLiNGAM(grid, dim)
    data.pc_graphs.append(pc_matrix)
    data.fci_graphs.append(fci_matrix)
    data.ges_graphs.append(ges_matrix)
    data.lin_graphs.append(lin_matrix)



# Performs the updated portion of the algorithm for RQ1 where we analyze results of all 
# of the algorithms on their appropriate expected graph, the expected graph is calculated
# to be the "average" graph of all of the generated pc_graphs and is a quick way to gauge 
# result stability, (i.e.) if the algorithm is stable then all of the graphs should match 
# the expected one and the average for that alg will wind up being 1, if the average is
# anything less than 1 we have observed instability in the output.
def exp1Expected(data : DATA, dim):
    data.prop_pc = generateExpectedGraph(data.pc_graphs, dim)
    data.prop_fci = generateExpectedGraph(data.fci_graphs, dim)
    data.prop_ges = generateExpectedGraph(data.ges_graphs, dim)
    data.prop_lin = generateExpectedGraph(data.lin_graphs, dim)
    pc_avg = listJaccards(data.pc_graphs, data.prop_pc, dim)
    fci_avg = listJaccards(data.fci_graphs, data.prop_fci, dim)
    ges_avg = listJaccards(data.ges_graphs, data.prop_ges, dim)
    lin_avg = listJaccards(data.lin_graphs, data.prop_lin, dim)
    data.pc = pc_avg
    data.fci = fci_avg
    data.ges = ges_avg
    data.lin = lin_avg
    print('EXPECTED: ', pc_avg, fci_avg, ges_avg, lin_avg)

# Gets the p_value from the hypothesis test associated with the Shapiro-Wilk test for every col, and returns the list
# Ignores the test statistic as none of my datasets were too large for it and I only really need to know if I should/can
# reject the null hypothesis of a normal distribution which can be found in this value
def getDistriution(df):
    dist = []
    count = 0
    for col in df.columns:
        _, p_value = stats.shapiro(df[col])
        dist.append(p_value)
    return dist

# Quick-ish automated check for confounders on unknown datasets, does not make any assumptions
# if we locate a moderate correlation (abs .5 or higher) between any potential pair of independent and 
# dependent variables then it is possible we have confounders, if this returns false it is supremely unlikely that we have these
# so true doesn't neccesarily mean we have them, but false should mean we do not in most cases, or at least their effect is minimal
def heuristic_check_for_confounders(df, threshold=.5):
    c_matrix = df.corr()
    for y in df.columns:
        for x in df.columns:
            if x != y:
                x_c = c_matrix[x]
                y_c = c_matrix[y]
                for col in df.columns:
                    if col != x and col != y:
                        if abs(x_c[col]) > threshold and  abs(y_c[col]) > threshold:
                            return True
    return False

            
# Selects the most stable algorithm that has been encountered by DATA in its experiments
# 1 = PC
# 2 = FCI
# 3 = GES
# 4 = LIN
# Does not currently support returning multiple values, so a potential enhancement could
# be handling ties here
def most_stable_alg(data : DATA):
    max = 1
    max_seen = data.pc
    if max_seen < data.fci:
        max = 2
        max_seen = data.fci
    if max_seen < data.ges:
        max = 3
        max_seen = data.ges
    if max_seen < data.lin:
        max = 4
        max_seen = data.lin
    return max, max_seen


# heuristic estimation based on the distribution of individual columns
# where we locate the average p_value for the hypothesis test conjured by the
# Shapiro-Wilk test, if the average is less than .05 and no columns have largely
# exceeded that (by 3 * std or more) then we say that the data may be normally distributed
# if that is not true we assume that the dataset is not normally distributed
def estimateNormal(df):
    distributions = getDistriution(df)
    mean = np.mean(distributions)
    std = np.std(distributions)
    max = np.std(distributions)
    return  mean < .05 and (max <= mean + (3 * std) or max <= .05)

# performs the analysis for RQ2, essentially analyzes each DATA, and 
# subsample in order to place it into the PROPERTY table for eventual prediction
# additionally has overlap with the REAL exps so there is alternate behavior when the exp2
# param is set to false where the data is added to the temp table and only
# the X cols are labeled, while the Y col remains unlabeled so we can label those later to assess
# accuracy on real_world sets
def analyze_sub_exp2(df, data: DATA, prop: PROPERTY, exp2 = True):
    approx_normal = estimateNormal(df)
    if exp2:
        exp1Expected(data, df.shape[1])
        max, _ = most_stable_alg(data)
        prop.props.append(( approx_normal, data.potential_conf, df.shape[0], max))
    else:
        prop.temp_real.append((approx_normal, data.potential_conf, df.shape[0]))



# performs the algorithm for RQ4, where we go through the unlabeled data in the temp space
# of PROPERTY and we begin guessing and checking our guesses on this data, once we make a guess
# and check it we upload the correct max to the PROPERTY table so this process will become more accurate
# with every guess made, even if it is wrong
def exp4(data: REAL, prop: PROPERTY):
    count = 0
    for row in prop.temp_real:
        sub = data.subsamples[count]
        count += 1
        alg = assign_likely_by_knn(prop, row[0], row[1], row[2])
        check_prediction(prop, sub, alg, row[0], row[1], row[2], data, False)

    prop.temp_real = []
    return 

def assign_likely_by_knn(prop: PROPERTY, approx_normal, potential_conf, rows, k =5):
    all = []
    similarities : List[float] = []
    for row in prop.props:
        sim :float  = 0
        sim += .33 if approx_normal == row[0] else 0
        sim += .33 if potential_conf == row[1] else 0
        sim += (.33 * (1 / (1 + abs(rows - row[2]))))
        idx = bisect.bisect(similarities, sim)
        all.insert(idx, row)
        similarities.insert(idx, sim)
    mostLike = all[-k:]
    freq = np.array([0, 0, 0, 0])
    for row in mostLike:
        freq[row[3] - 1] += 1
    return np.argmax(freq) + 1


def find_maximal_similar(data: DATA, grid: ndarray):
    dim = grid.shape[1]
    pc_matrix = performPC(grid, dim)
    fci_matrix = performFCI(grid, dim)
    ges_matrix = performGES(grid, dim)
    lin_matrix = performLiNGAM(grid, dim)
    pc_jaccard = calculateJaccardIndex(pc_matrix, data.prop_pc, dim)
    fci_jaccard = calculateJaccardIndex(fci_matrix, data.prop_fci, dim)
    ges_jaccard = calculateJaccardIndex(ges_matrix, data.prop_ges, dim)
    lin_jaccard = calculateJaccardIndex(lin_matrix, data.prop_lin, dim)
    max_seen = pc_jaccard
    max = 1
    if fci_jaccard > max_seen:
        max = 2
        max_seen = fci_jaccard
    if ges_jaccard > max_seen:
        max = 3
        max_seen = ges_jaccard
    if lin_jaccard > max_seen:
        max = 4
        max_seen = lin_jaccard
    return max

    
def check_prediction(prop: PROPERTY, sub: DataFrame,  pred, approx_normal, potential_conf, rows, data:DATA, synth = True):
    grid = sub.to_numpy()
    actual = find_maximal_similar(data, grid)
    improvement = data.getImprovement(pred)
    if actual == pred:
        if synth:
            prop.correct += 1
        else:
            prop.correct_real += 1
    if data.compareToMax(pred):
        if synth:
            prop.close += 1
        else:
            prop.close_real += 1
    if synth:
        prop.total += 1
        prop.imp += improvement
    else:
        prop.total_real += 1
        prop.imp_real += improvement
    prop.props.append((approx_normal, potential_conf, rows, actual))



    




# Hub method that performs experiments 1 and 2, these processes are semi-intermingled
# so they are done in this method
def run_experiments(files, N=30, p=.90, synthetic=True):
    results : List[DATA] = []
    property = PROPERTY()
    for file in files:
        df = pd.read_csv(file)
        
        data = SYNTHETIC(file) if synthetic else REAL(file)
        canon = pd.read_csv(file[:file.rfind('\\') + 1] + 'adj_matrix.csv', header=None)
        canon = canon.to_numpy()
        results.append(data)
        data.potential_conf = heuristic_check_for_confounders(df)
        for i in range(0, N):
            
            sub = df.sample(frac=p)
            np = sub.to_numpy()
            if data.type() == 'S':
                exp1(np, data, canon)
                analyze_sub_exp2(sub,data, property)
        
    count = 0
    for file in files:
         df = pd.read_csv(file)
         data = results[count]
         count += 1
         for i in range(0, N):
            sub = df.sample(frac=p - .1 if p -.1 > 0 else p / 2)
            approx_normal = estimateNormal(sub)
            conf = data.potential_conf
            pred = assign_likely_by_knn(property, approx_normal, conf, sub.shape[0] )
            check_prediction(property, sub, pred, approx_normal, conf, sub.shape[0], data )

    
    return results, property

def fix_sym_cols(df):
    for col in df.select_dtypes(include=['object']):
        df.loc[:,col] = pd.factorize(df[col])[0] + 1
    return df

def fix_invariance(df : DataFrame):
    return df.loc[:, df.nunique() > 1]

# Fixes a dataset for use in experiments, intended for real datasets
def fix_normal_data(df :DataFrame):
    df = fix_invariance(df)
    return df
    
def test_normal_data(file):
    df = pd.read_csv(file)
    df = fix_normal_data(df)
    file_name = 'test_' + file.rsplit('\\', 1)[-1]
    grid = df.to_numpy()
    print(grid)
    df.to_csv(file_name)


def run_real_experiments(files, results: List[DATA], prop : PROPERTY, N = 20, p = .9):
    results = []
    for file in files:
        data = REAL(file)
        df = pd.read_csv(file)
        results.append(data)
        fixed = fix_normal_data(df)
        data.potential_conf = heuristic_check_for_confounders(fixed)
        for i in range(N):
            sub = fixed.sample(frac=p)
            grid = sub.to_numpy()
            grid = np.array(grid, dtype=np.float64)
            analyze_sub_exp2(sub, data, prop, False)
            data.subsamples.append(sub)
            makeGraphs(grid, data)
        exp1Expected(data, fixed.shape[1])
        exp4(data, prop)

    return results, prop

def genExpectedGraphs(data: REAL, dim):
    return generateExpectedGraph(data.pc_graphs, dim), generateExpectedGraph(data.fci_graphs, dim), generateExpectedGraph(data.ges_graphs, dim), generateExpectedGraph(data.lin_graphs, dim)

def compareGraphs(t1, t2, dim):
    return calculateJaccardIndex(t1[0], t2[0], dim), calculateJaccardIndex(t1[1], t2[1], dim), calculateJaccardIndex(t1[2], t2[2], dim), calculateJaccardIndex(t1[3], t2[3], dim)

# Gets the jaccard values for all graphs in data made from the same generator
# Returns a tuple of list of jaccard indexes for each graph tyoe
# ([PC], [FCI], [GES], [LiNGAM])
def longJacc(data: REAL, dim) -> tuple[List, List, List, List]:
    results = ([],[],[],[])
    for i in range(len(data.pc_graphs)):
        for j in range(i + 1, len(data.pc_graphs)):
            g1 = (data.pc_graphs[i], data.fci_graphs[i], data.ges_graphs[i], data.lin_graphs[i])
            g2 = (data.pc_graphs[j], data.fci_graphs[j], data.ges_graphs[j], data.lin_graphs[j])
            pc, fci, ges, lin = (compareGraphs(g1, g2, dim))
            results[0].append(pc)
            results[1].append(fci)
            results[2].append(ges)
            results[3].append(lin)
           
    return results



# Performs the generative experiment meaning that we
# are testing our RQ0, for the increase in stability of the
# expected graph structure (we know this exists but we wish to quantify)
# Returns ([PC], [FCI], [GES], [LiNGAM])
def genEXP(df, file, N: int, p =.9) -> tuple[List, List, List, List]: 
    exp = REAL(file)
    for i in range(N):
        data = REAL(file)
        for i in range(N):
            sub = df.sample(frac=p)
            grid = sub.to_numpy()
            np.array(grid, dtype=np.float64)
            makeGraphs(grid, data)
        dim = grid.shape[1]
        expected = genExpectedGraphs(data, dim)
        exp.pc_graphs.append(expected[0])
        exp.fci_graphs.append(expected[1])
        exp.ges_graphs.append(expected[2])
        exp.lin_graphs.append(expected[3])
    return longJacc(exp, dim)
        
            


# Performs RQ0 for a specific file
def runRQ0(file, N=10, p=.9, compress=False):
    df = fix_normal_data(pd.read_csv(file))
    if compress:
        df = df.sample(100)
    pc, fci, ges, lin = genEXP(df, file, N, p)
    return (statistics.mean(pc), statistics.stdev(pc)), (statistics.mean(fci), statistics.stdev(fci)), (statistics.mean(ges), statistics.stdev(ges)), (statistics.mean(lin), statistics.stdev(lin))
        
def reportRQ0(tup, fileName):
    file = "results\\RQ0\\" + fileName[fileName.find('\\') + 1:]
    
    with open(file, 'a+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                print(tup[0][0], tup[0][1], tup[1][0], tup[1][1], tup[2][0], tup[2][1], tup[3][0], tup[3][1], sep=',', file=f, end='\n'  )
            finally:
                portalocker.unlock(f)


# SCRIPT PORTION
synth_files = get_files(initial_path, synethetic_pattern)
real_files = []
process = []
for path in proc_paths:
    proccess = get_files(path, real_pattern, found=process)


for path in real_paths:
    get_files(path, real_pattern, real_files)

# for file in real_files:
#     test_normal_data(file)

if sys.argv[1] == "-A":
    print("ALT MODE", flush=True)
    for file in proccess:
        reportRQ0(runRQ0(file, compress=True), file)
    for file in real_files:
        reportRQ0(runRQ0(file), file)
else:
    results, props = run_experiments(synth_files)
    for data in results:
        data.reportRQ1()
        data.reportRQ7()
    props.reportRQ2()
    real_results, prop = run_real_experiments(real_files, results, props)
    for data in real_results:
        data.reportRQ3()
        data.reportRQ8()
    props.reportRQ4()
    props.reportRQ5()
    props.reportRQ6()

