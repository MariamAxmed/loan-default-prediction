"""
Feature Engineering
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def cap_outliers(df):

    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

    numerical_columns = [
        col for col in numerical_columns
        if col != "SeriousDlqin2yrs"
    ]

    for column in numerical_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[column] = df[column].clip(lower, upper)

    return df


def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    return X_scaled, scaler