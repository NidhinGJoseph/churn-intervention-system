import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH


def load_raw_data(path=None):
    if path is None:
        path = RAW_DATA_PATH

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
