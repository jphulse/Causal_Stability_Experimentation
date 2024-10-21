# General imports for files
import os 
import re 

#Data imports
import pandas as pd
import numpy as np

# pip install causallearn
# causal learning imports
from causallearn.search.ConstraintBased import PC
from causallearn.search.ConstraintBased import FCI
from causallearn.search.ScoreBased import GES
from causallearn.search.FCMBased import lingam
from causallearn.utils.GraphUtils import GraphUtils

# data processing or modification imports
from sklearn.preprocessing import LabelEncoder

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

    def reportRQ1(self):
        print(self.file, self.count, self.pc, self.fci, self.ges, self.lin)
    
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

    
# class representing specific operations to be performed on synthetic data
class SYNTHETIC(DATA):

    def type(self):
        return 'S'

# Class representing specific operations to be performed on real datasets
class REAL(DATA):
    def type(self):
        return 'R'  





    

# INITIALIZATION, done first
initial_path = 'data\\'
real_paths = ['data\\ant', 'data\\ivy', 'data\\camel', 'data\\synapse', 'data\\xerces']
real_pattern = r'.*\.csv$'
synethetic_pattern = r'train.csv$'

# gets all of the files from the directory that match the provided regex pattern
def get_files(directory, regex, found=[], synthetic=True):
    reg = re.compile(regex)
    for path, _, files in os.walk(directory):
        for file in files:
            if reg.search(file):
                found.append(os.path.join(path, file))
    
    return found

# Takes a 2D edgelist and turns them into an adjacency matrix for comparisons
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
        if e1 == -1:
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
    pc_graph = PC.pc(grid)
    return turnEdgesToMatrix(pc_graph.find_adj(), dim)

# performs the FCI alg and returns the formatted matrix
def performFCI(grid, dim):
    _, fci_list = FCI.fci(grid)
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

# Performs the algorithm for RQ1
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


def exp1Expected(data : DATA, dim, canon):
    data.prop_pc = generateExpectedGraph(data.pc_graphs, dim)
    data.prop_fci = generateExpectedGraph(data.fci_graphs, dim)
    data.prop_ges = generateExpectedGraph(data.ges_graphs, dim)
    data.prop_lin = generateExpectedGraph(data.lin_graphs, dim)
    pc_avg = listJaccards(data.pc_graphs, data.prop_pc, dim)
    fci_avg = listJaccards(data.fci_graphs, data.prop_fci, dim)
    ges_avg = listJaccards(data.ges_graphs, data.prop_ges, dim)
    lin_avg = listJaccards(data.lin_graphs, data.prop_lin, dim)
    print('EXPECTED: ', pc_avg, fci_avg, ges_avg, lin_avg)
    print('EXPECTED_TO_CANON',calculateJaccardIndex(data.prop_pc, canon, dim), calculateJaccardIndex(data.prop_fci, canon, dim), calculateJaccardIndex(data.prop_ges, canon, dim), calculateJaccardIndex(data.prop_lin, canon, dim) )


# performs the algorithm for RQ2
def exp2(df):
    return 

# performs the algorithm for RQ3
def exp3(df):
    return 

# performs the algorithm for RQ4
def exp4(df):
    return 



# Hub method that performs all experiments N times for statistical validity and flexibility
def run_experiments(files, N=20, p=.9, synthetic=True):
    results = []
    for file in files:
        df = pd.read_csv(file)
        data = SYNTHETIC(file) if synthetic else REAL(file)
        canon = pd.read_csv(file[:file.rfind('\\') + 1] + 'adj_matrix.csv', header=None)
        canon = canon.to_numpy()
        results.append(data)
        dim = 0
        for i in range(0, N):
            
            sub = df.sample(frac=p)
            np = sub.to_numpy()
            dim = np.shape[1]
            if data.type() == 'S':
                exp1(np, data, canon)
                
        exp1Expected(data, dim, canon)

            
    
    return results


synth_files = get_files(initial_path, synethetic_pattern)
real_files = []
for path in real_paths:
    get_files(path, real_pattern, real_files)

results = run_experiments(synth_files, 20)
for data in results:
    data.reportRQ1()


