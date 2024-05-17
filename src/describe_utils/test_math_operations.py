from math_operations import operations
import pandas as pd
import numpy as np

def expectSysExit(funk, str):
    try:
        funk(str)
    except SystemExit:
        print("sys.exit() worked as expected")
        return
    print("Error exit failed!")
    exit(1)

def test_count():
    d = {'col1': [7, 1, 2, 3, 4], 'col2': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data=d)
    calc = operations(df)
    assert calc.count('col1') == 5
    assert calc.count('col2') == 5
    expectSysExit(calc.count, '111')


def test_mean():
    d = {'col1': [7, 1, 2, 3, 4, 1], 'col2': [1, 2, 3, 4, 77, 5]}
    df = pd.DataFrame(data=d)
    calc = operations(df)
    assert calc.mean('col1') == np.mean(d['col1'])
    assert calc.mean('col2') == np.mean(d['col2'])
    expectSysExit(calc.mean, '111')
