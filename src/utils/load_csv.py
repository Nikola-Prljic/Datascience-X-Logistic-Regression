import pandas as pd
import sys

def load(path: str):
    try:
        assert isinstance(path, str), "path is not str"
        df = pd.read_csv(path)
        return df
    except PermissionError as msg:
        print(f"{PermissionError.__name__}:\n{msg}")
    except FileNotFoundError as msg:
        print(f"{FileNotFoundError.__name__}:\n{msg}")
    except AssertionError as msg:
        print(f"{AssertionError.__name__}:\n{msg}")
    except pd.errors.EmptyDataError as msg:
        print(f"{pd.errors.EmptyDataError}:\n{msg}")
    except KeyboardInterrupt:
        sys.exit(1)
    print("at load_csv.py load()")