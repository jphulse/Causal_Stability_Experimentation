import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def analyze_rq1():
    df = pd.read_csv("results\\RQ1.csv")

    plt.figure(figsize=(10, 6))

    x_positions = np.arange(len(df.columns))

    for idx, col in enumerate(df.columns):
        x_vals = np.full(df[col].shape, x_positions[idx]) + np.random.normal(0, 0.05, size=df[col].shape)
        plt.scatter(x_vals, df[col], label=col, alpha=0.6)

    plt.xticks(x_positions, df.columns, fontsize=12)
    plt.ylim(0, 1)  
    plt.ylabel('Jaccard index')
    plt.title('')
    plt.savefig("results\\RQ1.png", format="png")


def analyze_rq2():
    df = pd.read_csv("results\\RQ2.csv")
    plt.figure(figsize=(10, 6))

    x_positions = np.arange(len(df.columns))

    for idx, col in enumerate(df.columns):
        x_vals = np.full(df[col].shape, x_positions[idx]) + np.random.normal(0, 0.05, size=df[col].shape)
        plt.scatter(x_vals, df[col], label=col, alpha=0.6)

    plt.xticks(x_positions, df.columns, fontsize=12)
    plt.ylim(0, 1)  
    plt.ylabel('Percentage')
    plt.title('')
    plt.savefig("results\\RQ2.png", format="png")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
  

def analyze_rq3():
    df = pd.read_csv("results\\RQ3.csv")

    plt.figure(figsize=(10, 6))

    x_positions = np.arange(len(df.columns))

    for idx, col in enumerate(df.columns):
        x_vals = np.full(df[col].shape, x_positions[idx]) + np.random.normal(0, 0.05, size=df[col].shape)
        plt.scatter(x_vals, df[col], label=col, alpha=0.6)

    plt.xticks(x_positions, df.columns, fontsize=12)
    plt.ylim(0, 1)  
    plt.ylabel('Jaccard index')
    plt.title('')
    plt.savefig("results\\RQ3.png", format="png")

def analyze_rq4():
    df = pd.read_csv("results\\RQ4.csv")
    plt.figure(figsize=(10, 6))

    x_positions = np.arange(len(df.columns))

    for idx, col in enumerate(df.columns):
        x_vals = np.full(df[col].shape, x_positions[idx]) + np.random.normal(0, 0.05, size=df[col].shape)
        plt.scatter(x_vals, df[col], label=col, alpha=0.6)

    plt.xticks(x_positions, df.columns, fontsize=12)
    plt.ylim(0, 1)  
    plt.ylabel('Percentage')
    plt.title('')
    plt.savefig("results\\RQ4.png", format="png")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

def analyze_5_6(csv_path='results/RQ5.csv', output_path='results/RQ5.png'):
        
    df = pd.read_csv(csv_path, header=None)
    
    
    x = df.index
    y = df[0].values  
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 1)
    plt.scatter(x, y, color='blue', marker='o')  
    plt.title('')
    plt.xlabel('Attempt')
    plt.ylabel('Average improvement of treatment (percentage based)')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  

analyze_rq1()
analyze_rq2()
analyze_rq3()
analyze_rq4()
analyze_5_6()
analyze_5_6('results/RQ6.csv', 'results/RQ6.png')