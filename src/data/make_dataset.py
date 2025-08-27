import pandas as pd
from pathlib import Path

def load_data(data_path):
    """
    Loads a dataset from the given path.

    Parameters
    ----------
    data_path : str or pathlib.Path
        The explicit path to the dataset file.
    
    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame.
    """
    full_path = Path(data_path)
    
    try:
        df = pd.read_csv(full_path)
        print(f"Dataset loaded successfully from {full_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {full_path}")
        print("Please ensure the path is correct and the file exists.")
        return None
