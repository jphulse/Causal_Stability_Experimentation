import pandas as pd
import matplotlib.pyplot as plt
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
    plt.title('Average stability of causal inference algorithms on synthetic datasets with small feature counts')
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
    plt.title('Percentage of correct and within 5% of the correct algorithm for algorithm selection for synthetic data')
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
    plt.title('Average stability of causal inference algorithms on real SE datasets with large feature counts')
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
    plt.title('Percentage of correct and within 5% of the correct algorithm for algorithm selection for real SE data')
    plt.savefig("results\\RQ4.png", format="png")
    plt.grid(axis='y', linestyle='--', alpha=0.7)



analyze_rq1()
analyze_rq2()
analyze_rq3()
analyze_rq4()