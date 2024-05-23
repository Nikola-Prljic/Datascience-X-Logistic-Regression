import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sys import argv
from utils.load_csv import load
from utils.python_colors import color

def standardize_df(df: pd.DataFrame):
    for col_name in df.columns:
        df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
    return df

def sim(df: pd.DataFrame):
    df = standardize_df(df)
    print(df)
    return df
    exit(1)
    

def plot_scatter(df: pd.DataFrame):
    #print(df[:10])
    #df = df[['Care of Magical Creatures', 'Defense Against the Dark Arts']]

    df = df.select_dtypes(include=['number'])
    df = df.dropna()
    df = sim(df)

    
    corr_df = df.corr()
    high_corr = corr_df[(corr_df > 0.8) & (corr_df < 1)]
    high_corr = high_corr.dropna(how='all')
    high_corr = high_corr.dropna(axis=1, how='all')
    #high_corr = high_corr[high_corr == True]
    print(high_corr)
    """ for columns_name in df.columns:
        print(columns_name)
        print(df[columns_name].mean(), df[columns_name].mean())
        print() """
        #Care of Magical Creatures
        #Defense Against the Dark Arts

    plt.scatter(df['Muggle Studies'], df['Charms'], label='class 1', color='green')
    plt.scatter(df['History of Magic'], df['Transfiguration'], label='class 2', color='red')
    plt.xlabel('Muggle Studies')
    plt.ylabel('Charms')
    plt.legend()
    plt.show()


def main():
    if len(argv) != 2:
        print(color('red', 'Error') + '\nNot right amounts of parameters\nTry python3 describe.py "filepath".csv')
        exit(1)
    df = load(argv[1])
    plot_scatter(df)
    

if __name__ == '__main__':
    main()