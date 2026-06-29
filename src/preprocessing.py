"""Data preprocessing utilities."""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT_DIR / "data" / "raw" / "cs-training.csv"

PROCESSED_DATA = ROOT_DIR / "data" / "processed"


def load_data():
    """Load raw dataset."""
    return pd.read_csv(RAW_DATA)


def remove_index_column(df):
    """Remove unnecessary index column."""

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def check_missing_values(df):
    """Return missing values."""

    return df.isnull().sum()


def fill_missing_values(df):
    """Fill missing values."""

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
        df["MonthlyIncome"].median()
    )

    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )

    return df


def split_features_target(df):

    X = df.drop("SeriousDlqin2yrs", axis=1)

    y = df["SeriousDlqin2yrs"]

    return X, y