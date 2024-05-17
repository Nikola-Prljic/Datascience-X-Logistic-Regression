import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sys import argv
from utils.load_csv import load
from utils.python_colors import color

def get_dicts():
    colors = {'Gryffindor': 'red',
              'Ravenclaw': 'blue',
              'Slytherin': 'green',
              'Hufflepuff': 'orange'}
    subjects_names = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                    'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                    'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    return colors, subjects_names

def variance(colum):
    var = sum(((colum - colum.mean()) ** 2))
    var = var / (len(colum) - 1)
    return var

def var_df(df: pd.DataFrame):
    variances = []
    for subject in df.columns:
        variances.append(variance(np.array(df[subject], dtype=np.float32)))
    variances = pd.DataFrame(variances, index=df.columns, columns=['variance'])
    return variances

def plt_histogram(df: pd.DataFrame):
    df = df.dropna()
    colors, subjects_names = get_dicts()
    #df[subjects_names] = df[subjects_names].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    variances = var_df(df[subjects_names])
    homogeneous_subject = variances.idxmin().iloc[0]
    plt.figure(figsize=(10, 6))

    for house in df['Hogwarts House'].unique():
        color = colors[house]
        plt.hist(df[df['Hogwarts House'] == house][homogeneous_subject], bins=10, alpha=0.4, label=house, color=color)

    plt.title(f'Histogram of Scores for {homogeneous_subject}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def plt_scatter(df: pd.DataFrame):
    df = df.dropna()
    score_columns = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    #df[score_columns] = df[score_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    variances = df[score_columns].var()
    homogeneous_subject = variances.idxmin()

    colors = {'Gryffindor': 'red',
              'Ravenclaw': 'blue',
              'Slytherin': 'green',
              'Hufflepuff': 'orange'}

    plt.figure(figsize=(12, 8))

    for house_name in df['Hogwarts House'].unique():
        color = colors[house_name]
        df_house = df[df['Hogwarts House'] == house_name][homogeneous_subject]
        plt.scatter(df_house, df_house.index, alpha=0.4, label=house_name, color=color)
    
    plt.title(f'Scatter of Scores for {homogeneous_subject}')
    plt.xlabel('Grade')
    plt.ylabel('Student Index')
    plt.legend()
    plt.show()

def main():
    if len(argv) != 2:
        print(color('red', 'Error') + '\nNot right amounts of parameters\nTry python3 describe.py "filepath".csv')
        exit(1)
    df = load(argv[1])
    plt_histogram(df)
    #plt_scatter(df)

if __name__ == "__main__":
    main()