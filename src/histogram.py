import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sys import argv
from utils.load_csv import load
from utils.python_colors import color

def plt_histogram(df: pd.DataFrame):
    df_ori = df.dropna()
    score_columns = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    df = df_ori.copy()
    df[score_columns] = df[score_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    variances = df[score_columns].var()
    homogeneous_subject = variances.idxmin()

    plt.figure(figsize=(10, 6))

    colors = {'Gryffindor': 'red',
              'Ravenclaw': 'blue',
              'Slytherin': 'green',
              'Hufflepuff': 'orange'}

    for house in df['Hogwarts House'].unique():
        color = colors[house]
        plt.hist(df[df['Hogwarts House'] == house][homogeneous_subject], bins=10, alpha=0.4, label=house, color=color)

    plt.title(f'Histogram of Scores for {homogeneous_subject}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    if len(argv) != 2:
        print(color('red', 'Error') + '\nNot right amounts of parameters\nTry python3 describe.py "filepath".csv')
        exit(1)
    df = load(argv[1])
    plt_histogram(df)

if __name__ == "__main__":
    main()