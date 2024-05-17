from sys import argv
from utils.load_csv import load
from utils.python_colors import color
from describe_utils.math_operations import operations
import pandas as pd
import numpy as np

def describe_csv(filepath: str):
    df = load(filepath)
    df = df.drop(columns=['Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'])
    df = df.dropna()
    calc = operations(df)
    features_dicts = {'Feature ' + str(i + 1) : calc.describe_col(f) 
                      for i, f in enumerate(df.columns)}
    df = pd.DataFrame(data=features_dicts,
                      index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    print(df)

def main():
    if len(argv) != 2:
        print(color('red', 'Error') + '\nNot right amounts of parameters\nTry python3 describe.py "filepath".csv')
        exit(1)
    describe_csv(filepath=argv[1])

if __name__ == "__main__":
    main()
