import pandas as pd
import numpy as np

class operations:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.dropna()

    def count(self, col_name: str) -> float:
        i = 0
        col = self.check_key(col_name)
        for row in col:
            i += 1
        return i

    def mean(self, col_name: str) -> float:
        length = self.count(col_name)
        return self.df[col_name].sum() / length

    def std(self, col_name: str) -> float:
        mean = self.mean(col_name)
        column = np.array(self.df[col_name])
        x = np.sum( (column - mean) ** 2 ) / ( self.count(col_name) )
        return np.sqrt(x)

    def min(self, col_name: str) -> float:
        col = self.check_key(col_name)
        col.sort()
        return col[0]
    
    def max(self, col_name: str) -> float:
        col = self.check_key(col_name)
        col.sort()
        return col[-1]

    def quantile(self, col_name, P):
        V = self.check_key(col_name)
        V_sorted = np.sort(V)
        n = self.count(col_name)
        p = P * (n - 1)
        i = int(np.floor(p))
        g = p - i
        i_adj = i
        if i_adj == n - 1:
            return V_sorted[-1]
        return (1 - g) * V_sorted[i] + g * V_sorted[i + 1]

    def check_key(self, col_name):
        try:
            col = np.array(self.df[col_name])
            return col
        except KeyError:
            print("KeyError\nKey not found in data frame. [operations class]")
            exit(1)

    def describe_col(self, col_name):
        feature = [self.count(col_name), self.mean(col_name), self.std(col_name), self.min(col_name),
                   self.quantile(col_name, 0.25), self.quantile(col_name, 0.50), self.quantile(col_name, 0.75),
                   self.max(col_name)]
        return feature
